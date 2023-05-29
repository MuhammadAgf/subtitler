import tkinter as tk
import threading
import signal
import sys
import time
from datetime import datetime, timedelta
from queue import Queue
from io import BytesIO

import speech_recognition as sr
from deep_translator import GoogleTranslator
import webrtcvad
import whisper
import torchaudio
import torch
import os

from tkinter import *

USE_ONNX = False
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=USE_ONNX)
(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils


class Subtitler:
    def __init__(self, offset_x, offset_y, font_size, color, bg_color, sacrificial_color, tk_timeout,
                 app_output_id, record_timeout, phrase_timeout, pause_threshold, model_type, vad_aggressiveness=1):
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.font_size = font_size
        self.color = color
        self.bg_color = bg_color
        self.sacrificial_color = sacrificial_color
        self.tk_timeout = tk_timeout

        self.app_output_id = app_output_id
        self.record_timeout = record_timeout
        self.phrase_timeout = phrase_timeout
        self.pause_threshold = pause_threshold
        self.model_type = model_type

        self.translation_queue = Queue()
        self.translator = GoogleTranslator(source='ja', target='en')
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(vad_aggressiveness)

    def translate_audio(self):
        data_queue = Queue()

        recorder = sr.Recognizer()
        recorder.pause_threshold = self.pause_threshold
        source = sr.Microphone(device_index=self.app_output_id)

        with source as m:
            recorder.adjust_for_ambient_noise(source)
            self.sample_rate = 16000
            print(f'Sample Rate: {self.sample_rate}')

        def record_callback(recorder, audio):
            data_queue.put((datetime.now(), audio.get_raw_data(convert_rate=self.sample_rate)))

        stop_listening = recorder.listen_in_background(source, record_callback, phrase_time_limit=self.record_timeout)

        phrase_time = None
        last_sample = bytes()
        while True:
            now = datetime.now()
            if not data_queue.empty():
                prev_phrase_time = phrase_time
                if phrase_time and now - phrase_time > timedelta(seconds=self.phrase_timeout):
                    last_sample = bytes()

                while not data_queue.empty():
                    phrase_time, data = data_queue.get()
                    if phrase_time and now - phrase_time > timedelta(seconds=self.phrase_timeout):
                        print(now, phrase_time, 'deleted')
                        continue
                    else:
                        print(now, phrase_time, 'used')
                        last_sample += data

                print(datetime.now(), 'start vad')
                audio_data = sr.AudioData(last_sample, self.sample_rate, 2)#source.SAMPLE_WIDTH)
                wav_data = BytesIO(audio_data.get_wav_data())
                with open('en_example.wav', 'w+b') as f:
                    f.write(wav_data.read())
                wav = read_audio('en_example.wav', sampling_rate=self.sample_rate)
                speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=self.sample_rate)
                if len(speech_timestamps) == 0:
                    time.sleep(0.05)
                    continue

                frames = collect_chunks(speech_timestamps, wav)
                print(len(frames))
                buffer_ = BytesIO()
                torchaudio.save(buffer_, collect_chunks(speech_timestamps, wav).unsqueeze(0), self.sample_rate, bits_per_sample=16, format='wav')
                audio_data = sr.AudioData(buffer_.getbuffer().tobytes(), self.sample_rate, 2)

                print(datetime.now(), 'done vad')
                text = recorder.recognize_whisper(audio_data, model=self.model_type, language='Japanese')
                print(datetime.now(), 'done whisper')

                try:
                    text += '\n' + self.translator.translate(text)
                except:
                    pass

                print(datetime.now(), 'done translation')
                phrase_time = now
                self.translation_queue.put(text)
            else:
                time.sleep(0.1)


    def subtitle_updater(self, root, label):
        while not self.translation_queue.empty():
            label.destroy()
            if root.wm_state() == 'withdrawn':
                root.deiconify()

            msg = self.translation_queue.get()
            label = tk.Label(
                text=msg,
                font=('Comic Sans MS', self.font_size, 'bold'),
                fg=self.color,
                bg=self.bg_color
            )

            label.after(self.tk_timeout, root.withdraw)
            label.after(self.tk_timeout, label.destroy)

            label.pack(side='bottom', anchor='s')
            root.update_idletasks()

        root.after(50, lambda: self.subtitle_updater(root, label))

    def setup_overlay(self, root=None):
        if root is None:
            root = tk.Tk()
        root.overrideredirect(True)
        root.geometry(f'{root.winfo_screenwidth()}x{root.winfo_screenheight()}+{self.offset_x}+{self.offset_y}')
        root.lift()
        root.wm_attributes('-topmost', True)
        root.wm_attributes('-transparentcolor', self.sacrificial_color)
        root.config(bg=self.sacrificial_color)

        return root

    def close_app(self, *_):
        print('Closing subtitler.')
        sys.exit(0)

    def start_app(self, root=None):
        signal.signal(signal.SIGINT, self.close_app)

        overlay = self.setup_overlay(root)
        subtitle = tk.Label()

        threading.Thread(target=self.translate_audio, daemon=True).start()

        self.subtitle_updater(overlay, subtitle)

        overlay.mainloop()

class SubtitlerApp(Frame):

    def __init__(self, master):
        self.root = master
        super().__init__(master)

        self.offset_x = Entry(self)
        self.offset_x.insert(0, 0)

        self.offset_y = Entry(self)
        self.offset_y.insert(0, -100)

        self.font_size = Entry(self)
        self.font_size.insert(0, 20)

        self.color = Entry(self)
        self.color.insert(0, "black")

        self.bg_color = Entry(self)
        self.bg_color.insert(0, "white")

        self.sacrificial_color = Entry(self)
        self.sacrificial_color.insert(0, "yellow")

        self.tk_timeout = Entry(self)
        self.tk_timeout.insert(0, 5000)

        self.app_output_id = StringVar(self)
        self.app_output_id.set("1")
        self.app_output_id_dropdown = OptionMenu(self, self.app_output_id, "1", "2", "3")

        self.record_timeout = Entry(self)
        self.record_timeout.insert(0, 3)

        self.phrase_timeout = Entry(self)
        self.phrase_timeout.insert(0, 3)

        self.pause_threshold = Entry(self)
        self.pause_threshold.insert(0, 0.75)

        self.model_type = Entry(self)
        self.model_type.insert(0, "base")

        self.start_button = Button(self, text="Start", command=self.start_app)

        self.pack()

        # Create labels for all the Entry widgets
        self.offset_x_label = Label(self, text="Offset X")
        self.offset_y_label = Label(self, text="Offset Y")
        self.font_size_label = Label(self, text="Font Size")
        self.color_label = Label(self, text="Color")
        self.bg_color_label = Label(self, text="Background Color")
        self.sacrificial_color_label = Label(self, text="Sacrificial Color")
        self.tk_timeout_label = Label(self, text="Tk Timeout")
        self.app_output_id_label = Label(self, text="App Output ID")
        self.record_timeout_label = Label(self, text="Record Timeout")
        self.phrase_timeout_label = Label(self, text="Phrase Timeout")
        self.pause_threshold_label = Label(self, text="Pause Threshold")
        self.model_type_label = Label(self, text="Model Type")

        # Grid the labels and Entry widgets
        self.offset_x_label.grid(row=0, column=1)
        self.offset_x.grid(row=0, column=2)
        self.offset_y_label.grid(row=1, column=1)
        self.offset_y.grid(row=1, column=2)
        self.font_size_label.grid(row=2, column=1)
        self.font_size.grid(row=2, column=2)
        self.color_label.grid(row=3, column=1)
        self.color.grid(row=3, column=2)
        self.bg_color_label.grid(row=4, column=1)
        self.bg_color.grid(row=4, column=2)
        self.sacrificial_color_label.grid(row=5, column=1)
        self.sacrificial_color.grid(row=5, column=2)
        self.tk_timeout_label.grid(row=6, column=1)
        self.tk_timeout.grid(row=6, column=2)
        self.app_output_id_label.grid(row=7, column=1)
        self.app_output_id_dropdown.grid(row=7, column=2)
        self.record_timeout_label.grid(row=8, column=1)
        self.record_timeout.grid(row=8, column=2)
        self.phrase_timeout_label.grid(row=9, column=1)
        self.phrase_timeout.grid(row=9, column=2)
        self.pause_threshold_label.grid(row=10, column=1)
        self.pause_threshold.grid(row=10, column=2)
        self.model_type_label.grid(row=11, column=1)
        self.model_type.grid(row=11, column=2)

        self.start_button = Button(self, text="Start", command=self.start_app)
        self.start_button.grid(row=12, column=0)

    def start_app(self):
        # Get the values from the user input
        offset_x = int(self.offset_x.get())
        offset_y = int(self.offset_y.get())
        font_size = int(self.font_size.get())
        color = self.color.get()
        bg_color = self.bg_color.get()
        sacrificial_color = self.sacrificial_color.get()
        tk_timeout = int(self.tk_timeout.get())
        app_output_id = int(self.app_output_id.get())
        record_timeout = int(self.record_timeout.get())
        phrase_timeout = int(self.phrase_timeout.get())
        pause_threshold = float(self.pause_threshold.get())
        model_type = self.model_type.get()

        subtitler = Subtitler(
            offset_x=offset_x,
            offset_y=offset_y,
            font_size=font_size,
            color=color,
            bg_color=bg_color,
            sacrificial_color=sacrificial_color,
            tk_timeout=tk_timeout,
            app_output_id=app_output_id,
            record_timeout=record_timeout,
            phrase_timeout=phrase_timeout,
            pause_threshold=pause_threshold,
            model_type=model_type
        )

        self.root.destroy()
        # Start the Subtitler app
        subtitler.start_app()

if __name__ == '__main__':
    root = Tk()
    app = SubtitlerApp(root)
    root.mainloop()
