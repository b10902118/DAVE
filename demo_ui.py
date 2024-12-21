# demo_ui.py
from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QGridLayout, QFileDialog, QDialog, QDialogButtonBox, QMessageBox, QScrollArea
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal  # Import pyqtSignal for creating custom signals
import os

class ImageSelectorUI(QMainWindow):
    # Declare a custom signal for when an image is selected
    image_selected = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        # Set up the main window properties
        self.setWindowTitle("Image Selector")
        self.setGeometry(100, 100, 800, 600)

        # Button to select a folder
        self.select_folder_button = QPushButton("Select Folder", self)

        # Layout to display images
        self.image_grid_layout = QGridLayout()

        # Scroll area for the image grid
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(QWidget())
        self.scroll_area.widget().setLayout(self.image_grid_layout)

        # Set up the main layout
        layout = QVBoxLayout()
        layout.addWidget(self.select_folder_button)
        layout.addWidget(self.scroll_area)

        # Set the layout to the main window
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Connect the button to the folder selection method
        self.select_folder_button.clicked.connect(self.load_images)

    def load_images(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.clear_image_grid()
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            for i, image_file in enumerate(image_files):
                image_path = os.path.join(folder_path, image_file)
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    # Create a QLabel for each image
                    image_label = QLabel()
                    image_label.setPixmap(pixmap.scaled(100, 100, aspectRatioMode=True))
                    image_label.setToolTip(image_path)  # Store the image path in the tooltip
                    image_label.mousePressEvent = lambda event, path=image_path: self.preview_image(path)
                    self.image_grid_layout.addWidget(image_label, i // 5, i % 5)

    def clear_image_grid(self):
        for i in reversed(range(self.image_grid_layout.count())): 
            widget = self.image_grid_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

    def preview_image(self, image_path):
        # Display a preview dialog when an image is clicked
        preview_dialog = ImagePreviewDialog(image_path)
        if preview_dialog.exec_() == QDialog.Accepted:
            print(f"Selected image: {image_path}")
            QMessageBox.information(self, "Selection Confirmed", f"You selected: {image_path}")
            # Emit the signal with the selected image path
            self.image_selected.emit(image_path)

class ImagePreviewDialog(QDialog):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("Image Preview")
        self.setGeometry(150, 150, 500, 350)

        # Create a label to display the image
        self.image_label = QLabel("Image Preview", self)
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.setText("Failed to load image")
        
        self.image_label.setScaledContents(True)
        self.image_label.setFixedSize(400, 300)

        # Create the confirm button
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        # Layout for the dialog
        dialog_layout = QVBoxLayout()
        dialog_layout.addWidget(self.image_label)
        dialog_layout.addWidget(self.button_box)
        self.setLayout(dialog_layout)
