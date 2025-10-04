# trainers/visualizer.py
"""
LiveTrainingVisualizer: Thread-safe, interrupt-safe live plot for training.

This callback plots:
- Loss
- Learning rate
- Optional grad norm

It handles:
- Automatic stopping when training ends
- Manual stop for testing or preflight
- Safe integration with HuggingFace Trainer callbacks
"""

import matplotlib.pyplot as plt
from threading import Thread
import time
from transformers import TrainerCallback

class LiveTrainingVisualizer(TrainerCallback):
    """
    HuggingFace TrainerCallback that visualizes training metrics in real-time.
    """
    def __init__(self):
        # Metrics storage
        self.epochs = []
        self.losses = []
        self.lrs = []
        self.grad_norms = []

        # Thread control
        self._stop_thread = False
        self._thread = Thread(target=self._plot_loop, daemon=True)
        self._thread.start()

    def _plot_loop(self):
        """
        Thread loop that continuously updates live plot.
        """
        plt.ion()
        self.fig, self.ax1 = plt.subplots(figsize=(10,5))
        self.ax2 = self.ax1.twinx()
        self.ax1.set_xlabel("Epoch")
        self.ax1.set_ylabel("Loss", color="#ff79c6")
        self.ax2.set_ylabel("LR", color="#8be9fd")
        self.ax1.tick_params(axis='y', labelcolor="#ff79c6")
        self.ax2.tick_params(axis='y', labelcolor="#8be9fd")
        self.ax1.set_title("🌸 Live Training Visualization 🌸", fontsize=14)

        while not self._stop_thread:
            if self.epochs:
                self.ax1.cla()
                self.ax2.cla()
                # Plot metrics
                self.ax1.plot(self.epochs, self.losses, color="#ff79c6", label="Loss")
                if any(self.grad_norms):
                    self.ax1.plot(self.epochs, self.grad_norms, color="#bd93f9", linestyle='--', label="Grad Norm")
                self.ax2.plot(self.epochs, self.lrs, color="#8be9fd", label="LR")

                # Labels and legends
                self.ax1.set_xlabel("Epoch")
                self.ax1.set_ylabel("Loss", color="#ff79c6")
                self.ax2.set_ylabel("LR", color="#8be9fd")
                self.ax1.tick_params(axis='y', labelcolor="#ff79c6")
                self.ax2.tick_params(axis='y', labelcolor="#8be9fd")
                self.ax1.legend(loc="upper left")
                self.ax2.legend(loc="upper right")

                try:
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                except Exception:
                    # Ignore GUI errors if training is interrupted
                    pass

            time.sleep(0.5)

    # ===== Trainer Callback Methods =====
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called by Trainer after every logging step."""
        if logs is None:
            return
        self.epochs.append(logs.get("epoch", len(self.epochs)))
        self.losses.append(logs.get("loss", 0))
        self.lrs.append(logs.get("learning_rate", 0))
        self.grad_norms.append(logs.get("grad_norm", 0))

    def on_train_end(self, args, state, control, **kwargs):
        """Stop plotting thread when training ends."""
        self._stop_thread = True
        if self._thread.is_alive():
            self._thread.join()

    # ===== Optional manual stop =====
    def stop(self):
        """Manually stop plotting thread (preflight/test)."""
        self._stop_thread = True
        if self._thread.is_alive():
            self._thread.join()
