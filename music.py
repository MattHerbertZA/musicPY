import tkinter as tk
from tkinter import filedialog, messagebox, Listbox, Scrollbar
import librosa
import librosa.display
import numpy as np
# import simpleaudio as sa # No longer needed
import sounddevice as sd
import threading
import os
import time # For potential small delays if needed
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# --- Configuration ---
MIN_LOOP_DURATION_SEC = 2.0
MAX_LOOP_DURATION_SEC = 15.0
SIMILARITY_THRESHOLD = 0.85
N_CANDIDATES = 15
FEATURE_TYPE = 'chroma'

# Global variable no longer needed for play object, using instance variables
# current_play_obj = None

# --- Core Loop Finding Logic ---
# find_potential_loops function remains the same as before
def find_potential_loops(y, sr, min_duration, max_duration, threshold, n_candidates, feature_type='chroma'):
    # --- Keep the exact same logic as in the previous working version ---
    # --- (Stopping playback is now handled by the app's stop_playback) ---
    # No changes needed inside this function itself unless you want to refine analysis
    # ... (rest of the find_potential_loops function code) ...

    try:
        print(f"Analyzing with sr={sr}, feature={feature_type}...")

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
            print("No initial loop pairs found above threshold/duration.")
            return []

        loop_pairs.sort(key=lambda x: x[2], reverse=True)

        # 4. Beat Tracking
        print("Performing beat tracking...")
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        # Ensure tempo is scalar before printing
        tempo_scalar = tempo.item() if isinstance(tempo, np.ndarray) else tempo
        print(f"Estimated Tempo: {tempo_scalar:.2f} BPM")
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

        if len(beat_times) < 2:
             print("Not enough beats detected for alignment. Using frame times.")
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
             print(f"Found {len(candidates)} potential frame-based loops.")
             return candidates

        # 5. Align Loop Candidates to Beats
        print("Aligning loops to beats...")
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
        print(f"Found {len(aligned_candidates)} potential aligned loops.")
        return aligned_candidates

    except Exception as e:
        print(f"Error during loop analysis: {e}")
        import traceback
        traceback.print_exc()
        return []
# --- GUI Application ---

class LoopFinderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Loop Finder")
        self.root.geometry("550x500") # Slightly wider for new button

        self.filepath = None
        self.y = None # Full audio data
        self.sr = None
        self.loop_candidates = []

        # Playback State Variables
        self.preview_thread = None
        self.loop_stream = None
        self.loop_data = None       # Holds the np.array for the segment being looped
        self.loop_current_frame = 0
        self.stop_event = threading.Event() # Used to signal loop callback/thread to stop
        self.playback_lock = threading.Lock() # Prevent starting multiple playbacks simultaneously

        # --- Top Frame: File Loading ---
        self.frame_top = tk.Frame(root, pady=5)
        self.frame_top.pack(fill=tk.X)

        self.btn_load = tk.Button(self.frame_top, text="Load Audio File", command=self.load_audio)
        self.btn_load.pack(side=tk.LEFT, padx=5)

        self.lbl_filename = tk.Label(self.frame_top, text="No file loaded.", anchor="w", fg="grey")
        self.lbl_filename.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # --- Middle Frame: Analysis and Loop List ---
        self.frame_middle = tk.Frame(root, pady=5)
        self.frame_middle.pack(fill=tk.BOTH, expand=True)


        self.lbl_loops = tk.Label(self.frame_middle, text="Potential Loop Points:")
        self.lbl_loops.pack(anchor="w", padx=5)

        self.listbox_frame = tk.Frame(self.frame_middle)
        self.listbox_frame.pack(fill=tk.BOTH, expand=True, padx=5)

        self.scrollbar = Scrollbar(self.listbox_frame, orient=tk.VERTICAL)
        self.listbox_loops = Listbox(self.listbox_frame, yscrollcommand=self.scrollbar.set, height=10)
        self.scrollbar.config(command=self.listbox_loops.yview)

        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox_loops.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox_loops.bind('<<ListboxSelect>>', self.on_loop_select)

        # --- Bottom Frame: Playback Controls and Status ---
        self.frame_bottom = tk.Frame(root, pady=5)
        self.frame_bottom.pack(fill=tk.X)

        # Renamed button for clarity
        self.btn_preview = tk.Button(self.frame_bottom, text="Preview Segment", command=self.preview_selected_segment, state=tk.DISABLED)
        self.btn_preview.pack(side=tk.LEFT, padx=5)

        # Button for looping playback
        self.btn_play_loop = tk.Button(self.frame_bottom, text="Play Loop", command=self.start_looping_playback, state=tk.DISABLED)
        self.btn_play_loop.pack(side=tk.LEFT, padx=5)

        # Stop button handles both
        self.btn_stop = tk.Button(self.frame_bottom, text="Stop Playback", command=self.stop_playback)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        # Save with loop button
        self.btn_save_with_loop = tk.Button(self.frame_bottom, text="Insert Loop & Save Song", 
                                          command=self.save_song_with_loop, state=tk.DISABLED)
        self.btn_save_with_loop.pack(side=tk.LEFT, padx=5)

        # --- Waveform Frame ---
        self.frame_waveform = tk.Frame(root, pady=5)
        self.frame_waveform.pack(fill=tk.X, padx=5)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(5, 1.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_waveform)
        self.canvas.get_tk_widget().pack(fill=tk.X, expand=True)
        
        # Store the current highlight patch
        self.highlight_patch = None

        self.lbl_status = tk.Label(root, text="Status: Idle", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.lbl_status.pack(side=tk.BOTTOM, fill=tk.X)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)


    def set_status(self, text):
        # Ensure status updates happen in the main GUI thread
        def update_label():
            self.lbl_status.config(text=f"Status: {text}")
        if self.root: # Check if root window still exists
            try:
                 self.root.after(0, update_label)
            except tk.TclError:
                 print("Status update failed (window closing?).")


    def load_audio(self):
        self.stop_playback() # Stop any playback before loading
        self.filepath = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.ogg *.flac"), ("All Files", "*.*")]
        )
        if not self.filepath:
            return

        self.lbl_filename.config(text=os.path.basename(self.filepath), fg="black")
        self.set_status(f"Loading {os.path.basename(self.filepath)}...")
        self.listbox_loops.delete(0, tk.END)
        self.btn_preview.config(state=tk.DISABLED)
        self.btn_play_loop.config(state=tk.DISABLED) # Disable loop button too
        self.btn_save_with_loop.config(state=tk.DISABLED)
        self.loop_candidates = []
        self.y = None
        self.sr = None

        try:
            threading.Thread(target=self._load_audio_thread, daemon=True).start()
        except Exception as e:
             messagebox.showerror("Load Error", f"Failed to load audio file:\n{e}")
             self.set_status("Error loading file.")
             self.lbl_filename.config(text="Load failed.", fg="red")


    def _load_audio_thread(self):
        try:
            self.y, self.sr = librosa.load(self.filepath, sr=None, mono=True)
             # Ensure data is float32 for sounddevice compatibility later
            if self.y.dtype != np.float32:
                self.y = self.y.astype(np.float32)

            # Update the waveform plot
            self.root.after(0, self.update_waveform_plot)
            
            self.set_status("Audio loaded successfully. Ready to analyze.")
            # automatically analyze the audio file
            self.run_analysis()
        except Exception as e:
            self.root.after(0, self.show_load_error, str(e))


    def show_load_error(self, error_message):
         messagebox.showerror("Load Error", f"Failed to load audio file:\n{error_message}")
         self.set_status("Error loading file.")
         self.lbl_filename.config(text="Load failed.", fg="red")


    def run_analysis(self):
        if self.y is None or self.sr is None:
            messagebox.showwarning("Analysis Warning", "Please load an audio file first.")
            return

        self.stop_playback() # Stop playback before analysis
        self.set_status("Analyzing audio for loops... (this may take a while)")
        self.btn_preview.config(state=tk.DISABLED)
        self.btn_play_loop.config(state=tk.DISABLED)
        self.btn_save_with_loop.config(state=tk.DISABLED)
        self.listbox_loops.delete(0, tk.END)

        analysis_thread = threading.Thread(
            target=self._analyze_thread,
            args=(self.y.copy(), self.sr), # Pass a copy of y
            daemon=True
        )
        analysis_thread.start()

    def _analyze_thread(self, y_data, sr_data):
        """Helper function to run analysis in background."""
        try:
            candidates = find_potential_loops(
                y_data, sr_data,
                MIN_LOOP_DURATION_SEC, MAX_LOOP_DURATION_SEC,
                SIMILARITY_THRESHOLD, N_CANDIDATES, FEATURE_TYPE
            )
            # Schedule GUI update in the main thread
            self.root.after(0, self.update_loop_list, candidates)
        except Exception as e:
             self.root.after(0, self.show_analysis_error, str(e))

    def update_loop_list(self, candidates):
        """Updates the listbox in the GUI thread."""
        self.loop_candidates = candidates # Store candidates
        self.listbox_loops.delete(0, tk.END)
        if self.loop_candidates:
            for i, (start, end, score) in enumerate(self.loop_candidates):
                duration = end - start
                list_text = f"{i+1: >2}. Loop: {start:>6.2f}s  ->  {end:>6.2f}s  (Duration: {duration:.2f}s)"
                self.listbox_loops.insert(tk.END, list_text)
            self.set_status(f"Found {len(self.loop_candidates)} potential loops. Select one to preview or play.")
            self.listbox_loops.select_set(0) # Select the first item
            self.on_loop_select(None) # Trigger button state update based on selection
        else:
            self.listbox_loops.insert(tk.END, "No suitable loops found.")
            self.set_status("Analysis complete. No loops found matching criteria.")
            self.btn_preview.config(state=tk.DISABLED) # Ensure buttons are disabled if no loops
            self.btn_play_loop.config(state=tk.DISABLED)
            self.btn_save_with_loop.config(state=tk.DISABLED)


    def show_analysis_error(self, error_message):
        """Displays analysis error in the GUI thread."""
        messagebox.showerror("Analysis Error", f"An error occurred during analysis:\n{error_message}")
        self.set_status("Analysis failed.")


    def on_loop_select(self, event):
        """Updates the waveform highlight when a loop is selected."""
        if self.listbox_loops.curselection() and self.loop_candidates:
            self.btn_preview.config(state=tk.NORMAL)
            self.btn_play_loop.config(state=tk.NORMAL)
            self.btn_save_with_loop.config(state=tk.NORMAL)
            
            # Update the waveform highlight
            start_time, end_time = self._get_selected_loop_times()
            if start_time is not None and end_time is not None:
                self.highlight_loop_region(start_time, end_time)
        else:
            self.btn_preview.config(state=tk.DISABLED)
            self.btn_play_loop.config(state=tk.DISABLED)
            self.btn_save_with_loop.config(state=tk.DISABLED)
            
            # Remove highlight if no selection
            if self.highlight_patch:
                self.highlight_patch.remove()
                self.highlight_patch = None
                self.canvas.draw()


    def _get_selected_loop_times(self):
        """Helper to get start/end times of the selected listbox item."""
        selected_indices = self.listbox_loops.curselection()
        if not selected_indices:
            # messagebox.showwarning("Playback Warning", "Please select a loop from the list.")
            return None, None

        selected_index = selected_indices[0]
        if selected_index >= len(self.loop_candidates):
             print("Error: Selected index out of bounds.")
             return None, None

        start_time, end_time, _ = self.loop_candidates[selected_index]
        return start_time, end_time

    # --- Playback Methods ---

    def preview_selected_segment(self):
        """Plays the selected segment ONCE."""
        start_time, end_time = self._get_selected_loop_times()
        if start_time is None:
            messagebox.showwarning("Playback Warning", "Please select a segment to preview.")
            return

        # Use lock to prevent concurrent playback starts
        if not self.playback_lock.acquire(blocking=False):
            print("Playback already in progress.")
            return

        try:
            # Don't call stop_playback() while holding the lock
            # Instead, use sd.stop() directly for preview
            sd.stop()  # Stop any current playback
            self.set_status(f"Previewing: {start_time:.2f}s - {end_time:.2f}s")

            # Run single playback in a separate thread
            self.preview_thread = threading.Thread(
                target=self._play_segment_once,
                args=(self.y, self.sr, start_time, end_time),
                daemon=True
            )
            self.preview_thread.start()
        finally:
            self.playback_lock.release()


    def _play_segment_once(self, y_data, sr_data, start_time, end_time):
        """Internal function to play a segment once using sounddevice."""
        try:
            start_sample = librosa.time_to_samples(start_time, sr=sr_data)
            end_sample = librosa.time_to_samples(end_time, sr=sr_data)
            segment = y_data[start_sample:end_sample]

            if segment.size == 0:
                print("Warning: Selected segment is empty.")
                self.set_status("Preview failed: Segment empty.")
                return

            # Ensure float32
            if segment.dtype != np.float32:
                 segment = segment.astype(np.float32)

            print(f"Playing segment once: {start_time:.2f}s - {end_time:.2f}s")
            sd.play(segment, samplerate=sr_data, blocking=True) # block=True waits here
            print("Preview finished.")
            self.set_status("Preview finished.")

        except sd.PortAudioError as pae:
            error_message = f"Sounddevice PortAudioError during preview: {pae}"
            print(error_message)
            self.set_status("Preview error.")
            # Optionally show messagebox via root.after from main thread if needed
            # self.root.after(0, lambda: messagebox.showerror("Preview Error", error_message))
        except Exception as e:
            error_message = f"Preview error: {e}"
            print(error_message)
            import traceback
            traceback.print_exc()
            self.set_status("Preview error.")
            # self.root.after(0, lambda: messagebox.showerror("Preview Error", error_message))


    def start_looping_playback(self):
        """Starts playing the selected segment repeatedly using a callback."""
        start_time, end_time = self._get_selected_loop_times()
        if start_time is None:
             messagebox.showwarning("Playback Warning", "Please select a loop to play.")
             return

        # Use lock to prevent concurrent playback starts
        if not self.playback_lock.acquire(blocking=False):
            print("Playback already in progress.")
            # messagebox.showinfo("Playback Busy", "Another playback operation is already running.")
            return

        try:
            self.stop_playback() # Ensure any previous playback (preview or loop) is fully stopped

            self.set_status(f"Looping: {start_time:.2f}s - {end_time:.2f}s")

            # --- Prepare loop data ---
            start_sample = librosa.time_to_samples(start_time, sr=self.sr)
            end_sample = librosa.time_to_samples(end_time, sr=self.sr)
            self.loop_data = self.y[start_sample:end_sample]

            if self.loop_data.size == 0:
                print("Warning: Selected loop segment is empty.")
                self.set_status("Loop failed: Segment empty.")
                self.playback_lock.release() # Release lock as we are not starting stream
                return

            # Ensure float32
            if self.loop_data.dtype != np.float32:
                self.loop_data = self.loop_data.astype(np.float32)

            # Ensure mono
            if self.loop_data.ndim > 1:
                 print("Warning: Converting loop data to mono.")
                 self.loop_data = np.mean(self.loop_data, axis=1).astype(np.float32)


            self.loop_current_frame = 0
            self.stop_event.clear() # Ensure stop flag is reset

            # --- Start the OutputStream ---
            print(f"Starting loop stream for segment: {start_time:.2f}s - {end_time:.2f}s")
            self.loop_stream = sd.OutputStream(
                samplerate=self.sr,
                channels=1, # Mono
                callback=self.audio_callback,
                finished_callback=self.loop_finished_callback,
                blocksize=1024 # Adjust blocksize if needed (powers of 2 often good)
            )
            self.loop_stream.start()
            print("Loop stream started.")
            # The lock will be released when stop_playback is called or in loop_finished_callback

        except sd.PortAudioError as pae:
            error_message = f"Failed to start loop stream (PortAudioError): {pae}"
            print(error_message)
            messagebox.showerror("Loop Error", error_message)
            self.set_status("Loop error.")
            self.loop_stream = None # Ensure stream is None
            self.playback_lock.release() # Release lock on error
        except Exception as e:
            error_message = f"Failed to start loop stream: {e}"
            print(error_message)
            import traceback
            traceback.print_exc()
            messagebox.showerror("Loop Error", error_message)
            self.set_status("Loop error.")
            self.loop_stream = None
            self.playback_lock.release()


    def audio_callback(self, outdata, frames, time, status):
        """Callback function for sounddevice OutputStream."""
        if status:
            print(f"Audio Callback Status Warning: {status}", flush=True)

        if self.stop_event.is_set() or self.loop_data is None:
            # Signal to stop or data is gone
            outdata[:] = 0 # Fill with silence
            print("Callback: Stop event set or no loop data. Raising CallbackStop.")
            raise sd.CallbackStop
        try:
            len_loop_data = len(self.loop_data)
            if len_loop_data == 0: # Should have been caught before starting stream, but check again
                outdata[:] = 0
                raise sd.CallbackStop

            # Calculate how much data to copy in this chunk
            remaining_in_loop = len_loop_data - self.loop_current_frame
            chunk1_size = min(frames, remaining_in_loop)

            # Copy the first part (or all if it fits)
            outdata[:chunk1_size, 0] = self.loop_data[self.loop_current_frame : self.loop_current_frame + chunk1_size]

            # Check if we need to wrap around (loop)
            if chunk1_size < frames:
                chunk2_size = frames - chunk1_size
                # Copy from the beginning of the loop data
                outdata[chunk1_size:, 0] = self.loop_data[:chunk2_size]
                self.loop_current_frame = chunk2_size # New position is start + chunk2_size
            else:
                self.loop_current_frame += chunk1_size
                # Important: Check if we landed exactly at the end, wrap around for next time
                if self.loop_current_frame >= len_loop_data:
                    self.loop_current_frame = 0

        except Exception as e:
            print(f"Error in audio callback: {e}", flush=True)
            # Optionally print traceback, but be careful in callback
            # import traceback
            # traceback.print_exc()
            outdata[:] = 0 # Silence on error
            raise sd.CallbackStop # Stop the stream on error


    def loop_finished_callback(self):
        """Called by sounddevice when the loop stream stops or is stopped."""
        print("Loop finished callback triggered.")
        self.loop_stream = None # Stream is implicitly closed or closing
        self.loop_data = None
        self.set_status("Playback stopped.")
        # Release the lock if it's still held (it should be)
        # Use try-release in case it was already released by an error path
        try:
            self.playback_lock.release()
            print("Playback lock released in finished_callback.")
        except threading.ThreadError:
            print("Playback lock was already released.")


    def stop_playback(self):
        """Stops any ongoing playback (preview or loop)."""
        print("Stop playback requested.")
        self.stop_event.set()  # Signal the callback thread to stop
        
        # Stop any non-callback playback (preview)
        sd.stop()

        # Stop and close the OutputStream if it's active
        if self.loop_stream is not None:
            try:
                print("Closing loop stream...")
                self.loop_stream.abort()
                self.loop_stream.close()
                print("Loop stream closed.")
            except Exception as e:
                print(f"Error closing loop stream: {e}")
            finally:
                self.loop_stream = None

        # Only try to release the lock if we own it
        try:
            if self.playback_lock.locked():
                self.playback_lock.release()
                print("Playback lock released in stop_playback.")
        except threading.ThreadError:
            pass  # Lock was already released

    def on_closing(self):
        """Called when the window is closed."""
        print("Closing application...")
        self.stop_playback()
        # Wait a tiny moment for stop actions to potentially complete
        time.sleep(0.1)
        self.root.destroy()

    def save_song_with_loop(self):
        """Inserts the selected loop multiple times and saves the modified song."""
        start_time, end_time = self._get_selected_loop_times()
        if start_time is None:
            messagebox.showwarning("Save Warning", "Please select a loop to insert.")
            return

        if self.filepath is None:
            messagebox.showerror("Save Error", "No source audio file loaded.")
            return

        try:
            # Ask user how many times to repeat the loop
            repeat_dialog = tk.Toplevel(self.root)
            repeat_dialog.title("Loop Repetitions")
            repeat_dialog.geometry("300x150")
            repeat_dialog.transient(self.root)  # Make dialog modal
            repeat_dialog.grab_set()

            tk.Label(repeat_dialog, 
                    text="How many times should the section repeat?",
                    pady=10).pack()

            repeat_var = tk.StringVar(value="4")  # Default value
            repeat_entry = tk.Entry(repeat_dialog, textvariable=repeat_var)
            repeat_entry.pack(pady=5)

            def process_and_save():
                try:
                    repeats = int(repeat_var.get())
                    if repeats < 1:
                        messagebox.showwarning("Invalid Input", "Please enter a positive number.")
                        return
                    repeat_dialog.destroy()
                    self._process_and_save_with_loop(start_time, end_time, repeats)
                except ValueError:
                    messagebox.showwarning("Invalid Input", "Please enter a valid number.")

            tk.Button(repeat_dialog, text="OK", command=process_and_save).pack(pady=10)
            tk.Button(repeat_dialog, text="Cancel", command=repeat_dialog.destroy).pack()

            # Center the dialog on the main window
            repeat_dialog.update_idletasks()
            x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (repeat_dialog.winfo_width() // 2)
            y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (repeat_dialog.winfo_height() // 2)
            repeat_dialog.geometry(f"+{x}+{y}")

        except Exception as e:
            error_message = f"Error preparing to save: {str(e)}"
            print(error_message)
            messagebox.showerror("Save Error", error_message)
            self.set_status("Error in save preparation.")

    def _process_and_save_with_loop(self, start_time, end_time, repeats):
        """Internal method to process the audio and save with repeated loop."""
        try:
            # Convert times to samples
            start_sample = librosa.time_to_samples(start_time, sr=self.sr)
            end_sample = librosa.time_to_samples(end_time, sr=self.sr)
            
            # Extract the loop segment
            loop_data = self.y[start_sample:end_sample]
            
            if loop_data.size == 0:
                messagebox.showerror("Save Error", "Selected loop segment is empty.")
                return

            # Create the new audio array
            # Keep everything before the loop
            new_audio = list(self.y[:start_sample])
            
            # Add the loop repeated times
            for _ in range(repeats):
                new_audio.extend(loop_data)
            
            # Add everything after the loop
            new_audio.extend(self.y[end_sample:])
            
            # Convert to numpy array, maintaining original data type
            new_audio = np.array(new_audio, dtype=self.y.dtype)

            # Generate new filename
            original_path = os.path.splitext(self.filepath)[0]
            original_ext = os.path.splitext(self.filepath)[1]
            new_filename = f"{original_path}_with_{repeats}x_loop{original_ext}"

            # Ask user for save location
            save_path = filedialog.asksaveasfilename(
                initialfile=os.path.basename(new_filename),
                defaultextension=original_ext,
                filetypes=[
                    ("WAV files", "*.wav"),
                    ("All files", "*.*")
                ]
            )

            if not save_path:  # User cancelled
                return

            # Save the modified song with highest quality settings
            self.set_status("Saving modified song...")
            sf.write(save_path, new_audio, self.sr, 'PCM_32')  # Using 32-bit PCM for highest quality

            self.set_status(f"Song saved successfully with {repeats} loop repetitions")
            messagebox.showinfo("Success", 
                              f"Song saved successfully with {repeats} loop repetitions to:\n{save_path}")

        except Exception as e:
            error_message = f"Error saving modified song: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            messagebox.showerror("Save Error", error_message)
            self.set_status("Error saving modified song.")

    def update_waveform_plot(self):
        """Updates the waveform visualization."""
        try:
            self.ax.clear()
            times = np.arange(len(self.y)) / self.sr
            self.ax.plot(times, self.y, color='blue', alpha=0.5, linewidth=0.5)
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Amplitude')
            self.ax.set_title('Waveform')
            # Remove margins to maximize waveform size
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            print(f"Error updating waveform: {e}")

    def highlight_loop_region(self, start_time, end_time):
        """Highlights the selected loop region on the waveform."""
        try:
            # Remove existing highlight if any
            if self.highlight_patch:
                self.highlight_patch.remove()
                self.highlight_patch = None

            # Add new highlight
            self.highlight_patch = self.ax.axvspan(
                start_time, end_time,
                color='red', alpha=0.3
            )
            self.canvas.draw()
        except Exception as e:
            print(f"Error highlighting region: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = LoopFinderApp(root)
    root.mainloop()