#!/usr/bin/env python
"""Small Windows-friendly UI for local VibeVoice-ASR-HF transcription."""

from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path
from tkinter import (
    BooleanVar,
    Button,
    Checkbutton,
    END,
    Entry,
    Frame,
    Label,
    Listbox,
    OptionMenu,
    StringVar,
    Tk,
    filedialog,
    messagebox,
    scrolledtext,
)

from transcribe_vibevoice_hf import (
    DEFAULT_MODEL_ID,
    configure_hf_cache,
    _safe_stem,
    _segments_to_srt,
    _select_device,
    _select_dtype,
    transcribe_one,
)


class TranscribeUi:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("VibeVoice ASR")
        self.root.geometry("820x680")
        self.root.minsize(760, 600)

        self.audio_files: list[str] = []
        self.worker: threading.Thread | None = None
        self.log_queue: "queue.Queue[str]" = queue.Queue()

        self.model_var = StringVar(value=DEFAULT_MODEL_ID)
        self.output_dir_var = StringVar(value=str(Path("transcripts")))
        self.device_var = StringVar(value="auto")
        self.dtype_var = StringVar(value="auto")
        self.prompt_var = StringVar(value="")
        self.write_srt_var = BooleanVar(value=True)
        self.offline_var = BooleanVar(value=False)

        self._build()
        self._poll_log_queue()

    def _build(self) -> None:
        outer = Frame(self.root, padx=12, pady=12)
        outer.pack(fill="both", expand=True)

        Label(outer, text="音声/動画ファイル").grid(row=0, column=0, sticky="w")
        file_buttons = Frame(outer)
        file_buttons.grid(row=0, column=1, sticky="e")
        Button(file_buttons, text="追加", command=self.add_files, width=10).pack(side="left", padx=(0, 6))
        Button(file_buttons, text="削除", command=self.remove_selected, width=10).pack(side="left", padx=(0, 6))
        Button(file_buttons, text="全消去", command=self.clear_files, width=10).pack(side="left")

        self.file_list = Listbox(outer, height=7)
        self.file_list.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(4, 10))

        Label(outer, text="出力フォルダ").grid(row=2, column=0, sticky="w")
        out_row = Frame(outer)
        out_row.grid(row=2, column=1, sticky="ew")
        Entry(out_row, textvariable=self.output_dir_var).pack(side="left", fill="x", expand=True)
        Button(out_row, text="選択", command=self.choose_output_dir, width=8).pack(side="left", padx=(6, 0))

        Label(outer, text="補助キーワード/文脈").grid(row=3, column=0, sticky="w", pady=(10, 0))
        Entry(outer, textvariable=self.prompt_var).grid(row=3, column=1, sticky="ew", pady=(10, 0))

        Label(outer, text="モデル").grid(row=4, column=0, sticky="w", pady=(10, 0))
        Entry(outer, textvariable=self.model_var).grid(row=4, column=1, sticky="ew", pady=(10, 0))

        options = Frame(outer)
        options.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(10, 10))
        Label(options, text="デバイス").pack(side="left")
        OptionMenu(options, self.device_var, "auto", "cuda", "cpu").pack(side="left", padx=(6, 18))
        Label(options, text="精度").pack(side="left")
        OptionMenu(options, self.dtype_var, "auto", "float32", "float16", "bfloat16").pack(side="left", padx=(6, 18))
        Checkbutton(options, text="SRTも出力", variable=self.write_srt_var).pack(side="left", padx=(0, 18))
        Checkbutton(options, text="オフライン", variable=self.offline_var).pack(side="left")

        action_row = Frame(outer)
        action_row.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.start_button = Button(action_row, text="文字起こし開始", command=self.start, height=2)
        self.start_button.pack(fill="x")

        Label(outer, text="ログ").grid(row=7, column=0, sticky="w")
        self.log_text = scrolledtext.ScrolledText(outer, height=16, state="disabled")
        self.log_text.grid(row=8, column=0, columnspan=2, sticky="nsew", pady=(4, 0))

        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(1, weight=1)
        outer.rowconfigure(8, weight=2)

    def add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="文字起こしする音声/動画を選択",
            filetypes=[
                ("Audio/Video", "*.wav *.mp3 *.m4a *.mp4 *.mov *.flac *.ogg *.aac *.webm *.mkv"),
                ("All files", "*.*"),
            ],
        )
        for path in paths:
            if path not in self.audio_files:
                self.audio_files.append(path)
                self.file_list.insert(END, path)

    def remove_selected(self) -> None:
        selected = list(self.file_list.curselection())
        selected.reverse()
        for index in selected:
            del self.audio_files[index]
            self.file_list.delete(index)

    def clear_files(self) -> None:
        self.audio_files.clear()
        self.file_list.delete(0, END)

    def choose_output_dir(self) -> None:
        path = filedialog.askdirectory(title="出力フォルダを選択")
        if path:
            self.output_dir_var.set(path)

    def log(self, message: str) -> None:
        self.log_queue.put(message)

    def _poll_log_queue(self) -> None:
        while True:
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_text.configure(state="normal")
            self.log_text.insert(END, message + "\n")
            self.log_text.see(END)
            self.log_text.configure(state="disabled")
        self.root.after(150, self._poll_log_queue)

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("実行中", "文字起こしは既に実行中です。")
            return
        if not self.audio_files:
            messagebox.showwarning("ファイル未選択", "音声/動画ファイルを追加してください。")
            return

        self.start_button.configure(state="disabled", text="実行中...")
        self.worker = threading.Thread(target=self._run_worker, daemon=True)
        self.worker.start()

    def _run_worker(self) -> None:
        try:
            cache_dir = configure_hf_cache()

            import torch
            from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

            model_id = self.model_var.get().strip() or DEFAULT_MODEL_ID
            output_dir = Path(self.output_dir_var.get().strip() or "transcripts")
            output_dir.mkdir(parents=True, exist_ok=True)
            prompt = self.prompt_var.get().strip() or None
            device = _select_device(torch, self.device_var.get())
            dtype = _select_dtype(torch, self.dtype_var.get(), device)
            offline = self.offline_var.get()

            self.log(f"Model: {model_id}")
            self.log(f"Hugging Face cache: {cache_dir}")
            self.log(f"Device: {device}")
            self.log(f"Dtype: {str(dtype).replace('torch.', '')}")
            if offline:
                self.log("Offline mode: enabled")
            self.log("モデルを読み込んでいます。初回はダウンロードに時間がかかります。")

            load_kwargs = {
                "torch_dtype": dtype,
                "local_files_only": offline,
            }
            if device == "cuda":
                load_kwargs["device_map"] = "auto"

            processor = AutoProcessor.from_pretrained(model_id, local_files_only=offline)
            model = VibeVoiceAsrForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
            if device == "cpu":
                model = model.to("cpu")
            model.eval()
            self.log("モデル読み込み完了。")

            for audio in self.audio_files:
                audio_path = Path(audio)
                result = transcribe_one(
                    processor=processor,
                    model=model,
                    torch=torch,
                    audio_path=audio_path,
                    prompt=prompt,
                    dtype=dtype,
                    max_new_tokens=32768,
                    tokenizer_chunk_size=None,
                )

                base = output_dir / _safe_stem(audio_path)
                txt_path = base.with_suffix(".txt")
                json_path = base.with_suffix(".json")

                txt_path.write_text(str(result["transcription"]).strip() + "\n", encoding="utf-8")
                import json

                json_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                self.log(f"出力: {txt_path}")
                self.log(f"出力: {json_path}")

                if self.write_srt_var.get():
                    srt_text = _segments_to_srt(result.get("segments"))
                    if srt_text:
                        srt_path = base.with_suffix(".srt")
                        srt_path.write_text(srt_text, encoding="utf-8")
                        self.log(f"出力: {srt_path}")

            self.log("完了しました。")
            self.root.after(0, lambda: messagebox.showinfo("完了", "文字起こしが完了しました。"))
        except Exception as exc:
            self.log("エラーが発生しました。")
            self.log(str(exc))
            if "1455" in str(exc) or "ページング ファイル" in str(exc):
                self.log("")
                self.log("対処: Windows の仮想メモリ/ページングファイルを増やしてください。")
                self.log("目安: 32768 MB 以上、余裕を見るなら 65536 MB。")
                self.log("設定後は Windows を再起動してから、もう一度実行してください。")
            self.log(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("エラー", str(exc)))
        finally:
            self.root.after(0, lambda: self.start_button.configure(state="normal", text="文字起こし開始"))


def main() -> None:
    root = Tk()
    TranscribeUi(root)
    root.mainloop()


if __name__ == "__main__":
    main()
