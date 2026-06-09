
import os,re,queue,threading,torch, timm
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image
from torchvision import transforms

MODEL_PATH = r"/efficientnet/model_en.pth"
MODEL_NAME = "efficientnet_b4"
IMG_SIZE   = 380

BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLEAN_TIMELINE = True

CLASSES = ['post', 'pre', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9+',
           'tB', 'tEB', 'tHB', 'tM', 'tPB2', 'tPNa', 'tPNf', 'tSB']

BIO_ORDER = ['pre', 'tPB2', 'tPNa', 'tPNf', 't2', 't3', 't4', 't5', 't6', 't7',
             't8', 't9+', 'tM', 'tSB', 'tB', 'tEB', 'tHB', 'post']

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

BG        = "#FFF1F7"   # fons finestra
CARD      = "#FCE4EF"   # panells
PINK      = "#F49AC2"   # botó primari
PINK_DK   = "#E573A3"   # accentuació
PINK_HEAD = "#F6C9DD"   # capçalera de la taula
ROW_ALT   = "#FCEAF3"   # fila alterna
ROW_SEL   = "#F4A9CB"   # fila seleccionada
TEXT      = "#7C5266"   # text principal
TEXT_SOFT = "#BC8FA6"   # text secundari
DISABLED  = "#ECD3DF"   # botó desactivat
WHITE     = "#FFFFFF"

_MODEL = None

def natural_key(filename):
    nums = re.findall(r'\d+', os.path.basename(filename))
    return [int(n) for n in nums] if nums else [float('inf')]

def list_images(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]
    files.sort(key=natural_key)
    return [os.path.join(folder, f) for f in files]

def load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    new_state = {}
    for k, v in state.items():
        k = k.replace("module.", "")
        k = k.replace("model.", "")
        new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if len(missing) > 5 or len(unexpected) > 5:
        raise RuntimeError(
            f"El checkpoint no encaixa amb {MODEL_NAME}.\n"
            f"Estats que falten: {len(missing)}, Estats inesperats: {len(unexpected)}.\n"
        )

    model.to(DEVICE).eval()
    _MODEL = model
    return model

@torch.no_grad()
def predict_folder(paths, progress_cb=None):
    model = load_model()
    preds = []
    n = len(paths)

    for start in range(0, n, BATCH_SIZE):
        batch_paths = paths[start:start + BATCH_SIZE]
        tensors = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            tensors.append(TRANSFORM(img))
        batch = torch.stack(tensors).to(DEVICE)

        out = model(batch)
        idx = torch.argmax(out, dim=1).cpu().tolist()
        preds.extend(CLASSES[i] for i in idx)

        if progress_cb:
            progress_cb(min(start + BATCH_SIZE, n), n)

    return preds


def majority_smooth(labels, window=5):
    if window <= 1 or not labels:
        return list(labels)
    half = window // 2
    n = len(labels)
    out = []
    for i in range(n):
        seg = labels[max(0, i - half):min(n, i + half + 1)]
        out.append(max(set(seg), key=seg.count))
    return out

def enforce_monotonic(labels):
    rank = {c: i for i, c in enumerate(BIO_ORDER)}
    out = []
    cur = -1
    for lab in labels:
        r = rank.get(lab, cur)
        if r < cur and out:
            out.append(out[-1])
        else:
            cur = max(cur, r)
            out.append(lab)
    return out

def build_ranges(labels):
    ranges = []
    if not labels:
        return ranges
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            ranges.append((labels[start], start + 1, i))
            start = i
    return ranges

class EmbryoApp:
    def __init__(self, root):
        self.root = root
        self.folder = None
        self.paths = []
        self.ranges = []
        self.per_frame = []
        self.q = queue.Queue()

        root.title("EmbryoAI")
        root.geometry("800x660")
        root.minsize(700, 560)
        root.configure(bg=BG)

        self._build_styles()

        header = ttk.Frame(root, style="TFrame", padding=(22, 18, 22, 6))
        header.pack(fill="x")
        ttk.Label(header, text="🌸EmbryoAI", style="Title.TLabel").pack(anchor="center")

        card = ttk.Frame(root, style="Card.TFrame", padding=18)
        card.pack(fill="x", padx=22, pady=10)

        row1 = ttk.Frame(card, style="Card.TFrame")
        row1.pack(fill="x")
        ttk.Button(row1, text="Seleccionar carpeta de l'embrió",
                   style="Pink.TButton", command=self.choose_folder).pack(side="left")
        self.folder_lbl = ttk.Label(row1, text="Cap carpeta seleccionada",
                                    style="CardSoft.TLabel")
        self.folder_lbl.pack(side="left", padx=12)

        row2 = ttk.Frame(card, style="Card.TFrame")
        row2.pack(fill="x", pady=(12, 0))
        self.run_btn = ttk.Button(row2, text="▶  Analitzar", style="Pink.TButton",
                                  command=self.start_analyze, state="disabled")
        self.run_btn.pack(side="left")
        self.export_btn = ttk.Button(row2, text="Exportar a Excel/CSV",
                                     style="Soft.TButton", command=self.export, state="disabled")
        self.export_btn.pack(side="left", padx=8)

        self.progress = ttk.Progressbar(root, mode="determinate",
                                       style="Pink.Horizontal.TProgressbar")
        self.progress.pack(fill="x", padx=24, pady=(4, 2))
        self.status = ttk.Label(root, text="", style="Status.TLabel")
        self.status.pack(fill="x", padx=24)

        table_frame = ttk.Frame(root, style="TFrame", padding=(22, 10, 22, 18))
        table_frame.pack(fill="both", expand=True)

        cols = ("fase", "inicio", "fin", "nframes")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", style="Pink.Treeview")
        for c, t, w in (("fase", "Fase", 130), ("inicio", "Frame inicial", 120),
                        ("fin", "Frame final", 120), ("nframes", "Nº frames", 120)):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="center")
        self.tree.tag_configure("odd", background=ROW_ALT)
        self.tree.tag_configure("even", background=WHITE)

        vsb = ttk.Scrollbar(table_frame, orient="vertical",
                            command=self.tree.yview, style="Pink.Vertical.TScrollbar")
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

    def _build_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)

        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=BG, foreground=PINK_DK,
                        font=("Segoe UI Semibold", 22, "bold"))
        style.configure("Sub.TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 11))
        style.configure("Tiny.TLabel", background=BG, foreground=TEXT_SOFT, font=("Segoe UI", 9))
        style.configure("CardSoft.TLabel", background=CARD, foreground=TEXT_SOFT, font=("Segoe UI", 10))
        style.configure("Status.TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 9))

        # Botó primari
        style.configure("Pink.TButton", background=PINK, foreground=WHITE,
                        font=("Segoe UI", 10, "bold"), borderwidth=0, relief="flat",
                        padding=(16, 9))
        style.map("Pink.TButton",
                  background=[("active", PINK_DK), ("disabled", DISABLED)],
                  foreground=[("disabled", "#C9AEBB")])

        # Botó secundari
        style.configure("Soft.TButton", background=CARD, foreground=PINK_DK,
                        font=("Segoe UI", 10, "bold"), borderwidth=0, relief="flat",
                        padding=(16, 9))
        style.map("Soft.TButton",
                  background=[("active", PINK_HEAD), ("disabled", BG)],
                  foreground=[("disabled", "#D3B7C4")])

        style.configure("Pink.Horizontal.TProgressbar", troughcolor=CARD,
                        background=PINK, borderwidth=0, thickness=12)

        style.configure("Pink.Treeview", background=WHITE, fieldbackground=WHITE,
                        foreground=TEXT, rowheight=32, font=("Segoe UI", 10),
                        borderwidth=0)
        style.configure("Pink.Treeview.Heading", background=PINK_HEAD, foreground=TEXT,
                        font=("Segoe UI", 10, "bold"), relief="flat", padding=8)
        style.map("Pink.Treeview.Heading", background=[("active", PINK_HEAD)])
        style.map("Pink.Treeview",
                  background=[("selected", ROW_SEL)],
                  foreground=[("selected", WHITE)])

        style.configure("Pink.Vertical.TScrollbar", background=PINK_HEAD,
                        troughcolor=BG, borderwidth=0, arrowcolor=PINK_DK)
        style.map("Pink.Vertical.TScrollbar", background=[("active", PINK)])

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Selecciona la carpeta de l'embrió")
        if not folder:
            return
        paths = list_images(folder)
        if not paths:
            messagebox.showwarning("Sense imatges",
                                   "Aquesta carpeta no conté imatges (.jpg/.png/...).")
            return
        self.folder = folder
        self.paths = paths
        self.folder_lbl.config(text=f"{os.path.basename(folder)}  ·  {len(paths)} frames",
                              foreground=TEXT)
        self.run_btn.config(state="normal")
        self.export_btn.config(state="disabled")
        for r in self.tree.get_children():
            self.tree.delete(r)
        self.status.config(text="Llest per analizar.")

    def start_analyze(self):
        self.run_btn.config(state="disabled")
        self.export_btn.config(state="disabled")
        self.progress.config(value=0, maximum=len(self.paths))
        self.status.config(text="Carregant el model i analitzant els frames…")
        for r in self.tree.get_children():
            self.tree.delete(r)

        t = threading.Thread(target=self._worker, daemon=True)
        t.start()
        self.root.after(100, self._poll)

    def _worker(self):
        try:
            def cb(done, total):
                self.q.put(("progress", done, total))
            raw = predict_folder(self.paths, progress_cb=cb)

            if CLEAN_TIMELINE:
                final = enforce_monotonic(majority_smooth(raw, window=5))
            else:
                final = raw

            ranges = build_ranges(final)
            per_frame = [
                (i + 1, os.path.basename(self.paths[i]), raw[i], final[i])
                for i in range(len(self.paths))
            ]
            self.q.put(("done", ranges, per_frame))
        except Exception as e:
            self.q.put(("error", str(e)))

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, done, total = msg
                    self.progress.config(value=done)
                    self.status.config(text=f"Analitzant…  {done}/{total} frames")
                elif kind == "done":
                    _, ranges, per_frame = msg
                    self.ranges = ranges
                    self.per_frame = per_frame
                    self._fill_table(ranges)
                    self.run_btn.config(state="normal")
                    self.export_btn.config(state="normal")
                    self.status.config(
                        text=f"Fet — {len(per_frame)} frames, {len(ranges)} fases detectades")
                    return
                elif kind == "error":
                    self.run_btn.config(state="normal")
                    self.status.config(text="Error durant l'análisis")
                    messagebox.showerror("Error", msg[1])
                    return
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _fill_table(self, ranges):
        for r in self.tree.get_children():
            self.tree.delete(r)
        for i, (lab, a, b) in enumerate(ranges):
            tag = "odd" if i % 2 else "even"
            self.tree.insert("", "end", values=(lab, a, b, b - a + 1), tags=(tag,))

    def export(self):
        if not self.ranges:
            return
        path = filedialog.asksaveasfilename(
            title="Guardar els resultats",
            defaultextension=".xlsx",
            initialfile=f"{os.path.basename(self.folder)}_morfocinetica.xlsx",
            filetypes=[("Excel", "*.xlsx"), ("CSV (resumen)", "*.csv")],
        )
        if not path:
            return
        try:
            import pandas as pd
            df_sum = pd.DataFrame(
                [(l, a, b, b - a + 1) for (l, a, b) in self.ranges],
                columns=["fase", "frame_inicio", "frame_fin", "n_frames"])
            df_frames = pd.DataFrame(
                self.per_frame,
                columns=["frame", "archivo", "pred_cruda", "pred_final"])

            if path.lower().endswith(".csv"):
                df_sum.to_csv(path, index=False)
            else:
                with pd.ExcelWriter(path, engine="openpyxl") as w:
                    df_sum.to_excel(w, sheet_name="resumen", index=False)
                    df_frames.to_excel(w, sheet_name="por_frame", index=False)
            messagebox.showinfo("Guardat", f"Resultats guardats a:\n{path}")
        except Exception as e:
            messagebox.showerror("Error a l'exportar", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    EmbryoApp(root)
    root.mainloop()