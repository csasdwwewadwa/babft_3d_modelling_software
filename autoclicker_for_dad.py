import sys
import time
import pyautogui
from pynput import keyboard

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QGridLayout, QLabel, QLineEdit, QComboBox, 
                             QRadioButton, QPushButton, QFrame, QButtonGroup)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIntValidator, QFont

# platform specific settings (L debugging)
if sys.platform == "darwin":
    pyautogui.PAUSE = 0.01 
else:
    pyautogui.PAUSE = 0
    pyautogui.MINIMUM_DURATION = 0
    pyautogui.MINIMUM_SLEEP = 0

class ClickerThread(QThread):
    """
    Background thread for the actual clicking logic.
    """
    finished_signal = pyqtSignal()

    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.is_running = True

    def run(self):
        try:
            # --- parse settings ---
            try:
                ms = float(self.settings['ms'] or 0) / 1000
                sec = float(self.settings['sec'] or 0)
                mins = float(self.settings['min'] or 0) * 60
            except ValueError:
                ms, sec, mins = 0.1, 0, 0
            
            interval = ms + sec + mins
            if interval <= 0.002:
                interval = 0.002

            button = self.settings['button']
            clicks = 2 if self.settings['type'] == "Double" else 1
            mode = self.settings['mode']
            limit = int(self.settings['limit'] or 0) if mode == "limit" else 0
            count = 0

            # --- clicking Loop ---
            while self.is_running:
                pyautogui.click(button=button, clicks=clicks)
                
                if mode == "limit":
                    count += 1
                    if count >= limit:
                        time.sleep(0.05) 
                        break
                
                # smart sleep (allows immediate interruption)
                end_time = time.time() + interval
                while time.time() < end_time and self.is_running:
                    remaining = end_time - time.time()
                    sleep_dur = min(remaining, 0.05)
                    if sleep_dur > 0:
                        time.sleep(sleep_dur)

        except Exception as e:
            print(f"Error in click thread: {e}")
        finally:
            # emit signal that logic is done
            self.finished_signal.emit()

    def stop(self):
        self.is_running = False

class ModernAutoClicker(QWidget):
    hotkey_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("autoclicker pro max plus 3000")
        self.setFixedSize(340, 310)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        
        self.is_clicking = False
        self.hotkey = keyboard.Key.f6
        self.worker_thread = None
        self.listener = None

        if sys.platform == "darwin":
            self.check_macos_permissions()
            
        self.setup_ui()
        self.apply_styles()
        self.setup_hotkey()

    def check_macos_permissions(self):
        print("--- macOS Check ---")
        print("Ensure 'Accessibility' and 'Input Monitoring' are enabled.")

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # interval
        self.create_section_label("Click Interval", main_layout)
        int_layout = QHBoxLayout()
        self.min_entry = self.create_unit_input("MIN", "0", int_layout)
        self.sec_entry = self.create_unit_input("SEC", "0", int_layout)
        self.ms_entry = self.create_unit_input("MS", "30", int_layout)
        main_layout.addLayout(int_layout)

        # options
        self.create_section_label("Click Options", main_layout)
        opt_layout = QGridLayout()
        
        opt_layout.addWidget(QLabel("Mouse Button"), 0, 0)
        self.btn_opt = QComboBox()
        self.btn_opt.addItems(["left", "right", "middle"])
        opt_layout.addWidget(self.btn_opt, 1, 0)

        opt_layout.addWidget(QLabel("Click Type"), 0, 1)
        self.type_opt = QComboBox()
        self.type_opt.addItems(["Single", "Double"])
        opt_layout.addWidget(self.type_opt, 1, 1)
        main_layout.addLayout(opt_layout)

        # repetition
        self.create_section_label("Repetition", main_layout)
        self.radio_group = QButtonGroup(self)
        
        self.radio_inf = QRadioButton("Repeat until stopped")
        self.radio_inf.setChecked(True)
        self.radio_group.addButton(self.radio_inf)
        main_layout.addWidget(self.radio_inf)

        limit_layout = QHBoxLayout()
        self.radio_limit = QRadioButton("Repeat limit:")
        self.radio_group.addButton(self.radio_limit)
        limit_layout.addWidget(self.radio_limit)
        
        self.limit_entry = QLineEdit("100")
        self.limit_entry.setFixedWidth(60)
        self.limit_entry.setValidator(QIntValidator())
        limit_layout.addWidget(self.limit_entry)
        limit_layout.addStretch()
        main_layout.addLayout(limit_layout)

        main_layout.addStretch()

        # actions
        action_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start (F6)")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self.start_clicking)
        self.start_btn.setObjectName("startBtn")
        
        self.stop_btn = QPushButton("Stop (F6)")
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.clicked.connect(self.stop_clicking)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setObjectName("stopBtn")

        action_layout.addWidget(self.start_btn)
        action_layout.addWidget(self.stop_btn)
        main_layout.addLayout(action_layout)

    def create_section_label(self, text, parent_layout):
        label = QLabel(text)
        font = QFont("Arial", 10)
        font.setBold(True)
        label.setFont(font)
        label.setStyleSheet("color: #888888;")
        parent_layout.addWidget(label)

    def create_unit_input(self, label_text, default, layout):
        container = QFrame()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(2)
        entry = QLineEdit(default)
        entry.setAlignment(Qt.AlignCenter)
        entry.setValidator(QIntValidator())
        entry.setFixedHeight(28)
        lbl = QLabel(label_text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("font-size: 10px; color: #aaaaaa;")
        vbox.addWidget(entry)
        vbox.addWidget(lbl)
        layout.addWidget(container)
        return entry

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget { background-color: #f0f0f0; color: #333333; font-family: Arial; font-size: 12px; }
            
            QLineEdit { background-color: #ffffff; border: 1px solid #cccccc; border-radius: 4px; color: #333333; padding: 2px; }
            QLineEdit:disabled { background-color: #e0e0e0; color: #888888; border: 1px solid #d0d0d0; }
            
            QComboBox { background-color: #ffffff; border: 1px solid #cccccc; border-radius: 4px; padding: 4px; color: #333333; }
            QComboBox::drop-down { border: none; }
            QComboBox:disabled { background-color: #e0e0e0; color: #888888; }
            
            QPushButton { border-radius: 4px; font-weight: bold; padding: 8px; font-size: 13px; }
            
            QPushButton#startBtn { background-color: #27ae60; color: white; }
            QPushButton#startBtn:hover { background-color: #2ecc71; }
            QPushButton#startBtn:disabled { background-color: #98dcb2; color: #f0f0f0; }
            
            QPushButton#stopBtn { background-color: #c0392b; color: white; }
            QPushButton#stopBtn:hover { background-color: #e74c3c; }
            QPushButton#stopBtn:disabled { background-color: #e6b0aa; color: #f0f0f0; }
            
            QRadioButton { spacing: 8px; color: #333333; }
            QRadioButton:disabled { color: #888888; }
        """)

    def setup_hotkey(self):
        self.hotkey_signal.connect(self.toggle_clicking)
        def on_press(key):
            if key == self.hotkey:
                self.hotkey_signal.emit()
        try:
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()
        except Exception as e:
            print(f"Error starting keyboard listener: {e}")

    def toggle_clicking(self):
        if self.is_clicking:
            self.stop_clicking()
        else:
            self.start_clicking()

    def start_clicking(self):
        if self.is_clicking:
            return

        self.is_clicking = True
        self.toggle_inputs(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        settings = {
            'min': self.min_entry.text(),
            'sec': self.sec_entry.text(),
            'ms': self.ms_entry.text(),
            'button': self.btn_opt.currentText(),
            'type': self.type_opt.currentText(),
            'mode': "limit" if self.radio_limit.isChecked() else "inf",
            'limit': self.limit_entry.text()
        }

        self.worker_thread = ClickerThread(settings)
        self.worker_thread.finished_signal.connect(self.on_worker_finished)
        self.worker_thread.start()

    def stop_clicking(self):
        if not self.is_clicking or not self.worker_thread:
            return
        self.worker_thread.stop()
        # NOTE: we rely on the thread finishing to trigger on_worker_finished

    def on_worker_finished(self):
        """
        Called when the thread emits finished_signal.
        This handles the cleanup SAFELY to prevent crashes.
        """
        if self.worker_thread is not None:
            self.worker_thread.wait()
        
        self.worker_thread = None
        self.is_clicking = False
        self.toggle_inputs(True)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def toggle_inputs(self, enabled):
        inputs = [self.min_entry, self.sec_entry, self.ms_entry, 
                  self.btn_opt, self.type_opt, self.limit_entry, 
                  self.radio_inf, self.radio_limit]
        for widget in inputs:
            widget.setEnabled(enabled)

    def closeEvent(self, event):
        if self.worker_thread:
            self.worker_thread.stop()
            self.worker_thread.wait()
        if self.listener:
            self.listener.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModernAutoClicker()
    window.show()
    sys.exit(app.exec_())