from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import librosa
import librosa.display
import numpy as np
from sse_starlette.sse import EventSourceResponse
import asyncio


app = FastAPI()

# CORS for frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL if deploying
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    
)

class LoopRequest(BaseModel):
    data: str

# Add a global queue for status messages
status_updates = asyncio.Queue()

# Modify the print statements in find_potential_loops to send status updates
async def send_status(message):
    await status_updates.put(message)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/find-loop")
async def find_loop(audio: UploadFile = File(...)):
    try:
        # Read the file content
        content = await audio.read()

        asyncio.create_task(send_status('Analyzing audio file...'))
        
        # Here you would process the audio file
        # For now, returning a simple response
        return {"result": f"Received audio file: {audio.filename}"}
    
    except Exception as e:
        return {"error": str(e)}

@app.get('/status-stream')
async def status_stream():
    async def event_generator():
        while True:
            if not status_updates.empty():
                message = await status_updates.get()
                yield {
                    "event": "message",
                    "data": message
                }
            await asyncio.sleep(0.1)
    
    return EventSourceResponse(event_generator())

def find_potential_loops(y, sr, min_duration, max_duration, threshold, n_candidates, feature_type='chroma'):
    try:
        asyncio.create_task(send_status(f"Analyzing with sr={sr}, feature={feature_type}..."))

        # 1. Feature Extraction
        hop_length = 512
        if feature_type == 'mfcc':
            features = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length)
        else: # Default to chroma
            features = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

        # 2. Self-Similarity Matrix
        R = librosa.segment.recurrence_matrix(features, mode='affinity', metric='cosine', sparse=True)

        # 3. Find high-similarity pairs
        min_loop_frames = int(librosa.time_to_frames(min_duration, sr=sr, hop_length=hop_length))
        max_loop_frames = int(librosa.time_to_frames(max_duration, sr=sr, hop_length=hop_length))

        i_upper, j_upper = np.triu_indices_from(R, k=min_loop_frames)
        similarities = np.array(R[i_upper, j_upper]).flatten()

        valid_indices = np.where(similarities >= threshold)[0]
        loop_pairs = []
        for idx in valid_indices:
            start_frame = i_upper[idx]
            end_frame = j_upper[idx]
            duration_frames = end_frame - start_frame
            if min_loop_frames <= duration_frames <= max_loop_frames:
                loop_pairs.append((start_frame, end_frame, similarities[idx]))

        if not loop_pairs:
            asyncio.create_task(send_status("No initial loop pairs found above threshold/duration."))
            return []

        loop_pairs.sort(key=lambda x: x[2], reverse=True)

        # 4. Beat Tracking
        asyncio.create_task(send_status("Performing beat tracking..."))
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        # Ensure tempo is scalar before printing
        tempo_scalar = tempo.item() if isinstance(tempo, np.ndarray) else tempo
        asyncio.create_task(send_status(f"Estimated Tempo: {tempo_scalar:.2f} BPM"))
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

        if len(beat_times) < 2:
             asyncio.create_task(send_status("Not enough beats detected for alignment. Using frame times."))
             candidates = []
             added_loops_time = set()
             for start_frame, end_frame, score in loop_pairs:
                 start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
                 end_time = librosa.frames_to_time(end_frame, sr=sr, hop_length=hop_length)
                 loop_key = (round(start_time, 2), round(end_time, 2)) # Key based on time

                 is_redundant = loop_key in added_loops_time
                 if not is_redundant:
                     for c_start, c_end, _ in candidates:
                          if abs(c_start - start_time) < 1.0 and abs(c_end - end_time) < 1.0:
                              is_redundant = True
                              break
                 if not is_redundant:
                     candidates.append((start_time, end_time, score))
                     added_loops_time.add(loop_key)
                     if len(candidates) >= n_candidates:
                            break
             candidates.sort(key=lambda x: x[0]) # Sort by start time
             asyncio.create_task(send_status(f"Found {len(candidates)} potential frame-based loops."))
             return candidates

        # 5. Align Loop Candidates to Beats
        asyncio.create_task(send_status("Aligning loops to beats..."))
        aligned_candidates = []
        added_loops_beats = set() # Keep track of added start/end beat pairs

        for start_frame, end_frame, score in loop_pairs:
            start_time_approx = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
            end_time_approx = librosa.frames_to_time(end_frame, sr=sr, hop_length=hop_length)

            start_beat_idx = np.argmin(np.abs(beat_times - start_time_approx))
            end_beat_idx = np.argmin(np.abs(beat_times - end_time_approx))

            if start_beat_idx < end_beat_idx:
                start_time_aligned = beat_times[start_beat_idx]
                end_time_aligned = beat_times[end_beat_idx]
                duration_aligned = end_time_aligned - start_time_aligned

                if min_duration <= duration_aligned <= max_duration:
                    loop_key = (start_beat_idx, end_beat_idx)
                    is_redundant = loop_key in added_loops_beats
                    if not is_redundant:
                        for added_start, added_end, _ in aligned_candidates:
                           overlap_start = max(start_time_aligned, added_start)
                           overlap_end = min(end_time_aligned, added_end)
                           overlap_duration = max(0, overlap_end - overlap_start)
                           if duration_aligned > 0 and overlap_duration / duration_aligned > 0.5 or \
                              (added_end - added_start) > 0 and overlap_duration / (added_end - added_start) > 0.5:
                               is_redundant = True
                               break

                    if not is_redundant:
                        aligned_candidates.append((start_time_aligned, end_time_aligned, score))
                        added_loops_beats.add(loop_key)
                        if len(aligned_candidates) >= n_candidates:
                            break

        aligned_candidates.sort(key=lambda x: x[0])
        asyncio.create_task(send_status(f"Found {len(aligned_candidates)} potential aligned loops."))
        return aligned_candidates

    except Exception as e:
        asyncio.create_task(send_status(f"Error during loop analysis: {e}"))
        import traceback
        traceback.print_exc()
        return []