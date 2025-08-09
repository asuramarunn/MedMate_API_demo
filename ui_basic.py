import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

class MedicalAgentUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Medical Multi-Agent Assistant")
        self.geometry("700x600")
        self.resizable(False, False)
        self.create_widgets()
        self.context = {
            "symptoms": "",
            "history": "",
            "user_differentials": "",
            "lab_orders": "",
            "lab_tests": "",
            "lab_results": "",
            "lab_test_results": "",
            "diagnoses": "",
            "final_diagnosis": "",
            "treatment_plan": ""
        }

    def create_widgets(self):
        # Notebook for steps
        self.notebook = ttk.Notebook(self)
        self.frames = {}
        steps = [
            ("1. Chẩn đoán sơ bộ", self.step1_frame),
            ("2. Đánh giá chỉ định xét nghiệm", self.step2_frame),
            ("3. Đánh giá kết quả xét nghiệm", self.step3_frame),
            ("4. Đánh giá chẩn đoán cuối", self.step4_frame),
            ("5. Đánh giá phác đồ điều trị", self.step5_frame)
        ]
        for name, func in steps:
            frame = ttk.Frame(self.notebook)
            func(frame)
            self.notebook.add(frame, text=name)
            self.frames[name] = frame
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def step1_frame(self, frame):
        self._add_label_entry(frame, "Triệu chứng:", "symptoms", 0)
        self._add_label_entry(frame, "Tiền sử bệnh:", "history", 1)
        self._add_label_entry(frame, "Chẩn đoán phân biệt:", "user_differentials", 2)
        btn = ttk.Button(frame, text="Chạy agent chẩn đoán", command=self.run_diagnosis_agent)
        btn.grid(row=3, column=0, columnspan=2, pady=10)
        self._add_output_box(frame, "output1", 4)

    def step2_frame(self, frame):
        self._add_label_entry(frame, "Chỉ định xét nghiệm:", "lab_orders", 0)
        self._add_label_entry(frame, "Xét nghiệm đã làm:", "lab_tests", 1)
        self._add_label_entry(frame, "Chẩn đoán mục tiêu:", "diagnoses", 2)
        btn = ttk.Button(frame, text="Chạy agent đánh giá xét nghiệm", command=self.run_lab_test_evaluation_agent)
        btn.grid(row=3, column=0, columnspan=2, pady=10)
        self._add_output_box(frame, "output2", 4)

    def step3_frame(self, frame):
        self._add_label_entry(frame, "Kết quả xét nghiệm (Tên: kết quả; ...):", "lab_results", 0)
        self._add_label_entry(frame, "Chẩn đoán mục tiêu:", "diagnoses", 1)
        btn = ttk.Button(frame, text="Chạy agent đánh giá kết quả xét nghiệm", command=self.run_lab_test_result_evaluation_agent)
        btn.grid(row=2, column=0, columnspan=2, pady=10)
        self._add_output_box(frame, "output3", 3)

    def step4_frame(self, frame):
        self._add_label_entry(frame, "Chẩn đoán cuối:", "final_diagnosis", 0)
        btn = ttk.Button(frame, text="Chạy agent đánh giá chẩn đoán cuối", command=self.not_implemented)
        btn.grid(row=1, column=0, columnspan=2, pady=10)
        self._add_output_box(frame, "output4", 2)

    def step5_frame(self, frame):
        self._add_label_entry(frame, "Phác đồ điều trị:", "treatment_plan", 0)
        btn = ttk.Button(frame, text="Chạy agent đánh giá phác đồ điều trị", command=self.not_implemented)
        btn.grid(row=1, column=0, columnspan=2, pady=10)
        self._add_output_box(frame, "output5", 2)

    def _add_label_entry(self, frame, label, key, row):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(frame, width=60)
        entry.grid(row=row, column=1, padx=5, pady=5)
        entry.insert(0, self.context.get(key, ""))
        setattr(self, f"entry_{key}", entry)

    def _add_output_box(self, frame, name, row):
        box = scrolledtext.ScrolledText(frame, width=80, height=10, state='disabled')
        box.grid(row=row, column=0, columnspan=2, padx=5, pady=5)
        setattr(self, name, box)

    def update_context(self):
        for key in self.context:
            entry = getattr(self, f"entry_{key}", None)
            if entry:
                self.context[key] = entry.get()

    def run_diagnosis_agent(self):
        self.update_context()
        # TODO: Gọi agent thực tế ở đây
        output = f"[Fake] Kết quả chẩn đoán cho: {self.context['symptoms']}\nChẩn đoán phân biệt: {self.context['user_differentials']}"
        self.output1.config(state='normal')
        self.output1.delete(1.0, tk.END)
        self.output1.insert(tk.END, output)
        self.output1.config(state='disabled')

    def run_lab_test_evaluation_agent(self):
        self.update_context()
        # TODO: Gọi agent thực tế ở đây
        output = f"[Fake] Đánh giá xét nghiệm cho: {self.context['lab_tests']}\nChẩn đoán mục tiêu: {self.context['diagnoses']}"
        self.output2.config(state='normal')
        self.output2.delete(1.0, tk.END)
        self.output2.insert(tk.END, output)
        self.output2.config(state='disabled')

    def run_lab_test_result_evaluation_agent(self):
        self.update_context()
        # TODO: Gọi agent thực tế ở đây
        output = f"[Fake] Đánh giá kết quả xét nghiệm: {self.context['lab_results']}\nChẩn đoán mục tiêu: {self.context['diagnoses']}"
        self.output3.config(state='normal')
        self.output3.delete(1.0, tk.END)
        self.output3.insert(tk.END, output)
        self.output3.config(state='disabled')

    def not_implemented(self):
        messagebox.showinfo("Thông báo", "Chức năng này chưa được triển khai.")

if __name__ == "__main__":
    app = MedicalAgentUI()
    app.mainloop()
