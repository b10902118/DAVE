# demo.py
import sys
import subprocess
from PyQt5.QtWidgets import QApplication, QMessageBox
from demo_ui import ImageSelectorUI

def main():
    app = QApplication(sys.argv)
    window = ImageSelectorUI()

    # Connect the image selection signal to the function that runs the target script
    window.image_selected.connect(run_target_script)

    window.show()
    sys.exit(app.exec_())

def run_target_script(image_path):
    try:
        # Call target.py with the selected image path
        result = subprocess.run(["python", "demo.py", '--skip_train', '--model_name', "DAVE_3_shot", "--model_path", 'material', '--backbone', 'resnet50', '--swav_backbone', '--reduction', '8', '--num_enc_layers', '3', '--num_dec_layers', '3', '--kernel_dim', '3', '--emb_dim', '256', '--num_objects', '3', '--num_workers', '16', '--use_query_pos_emb', '--use_objectness', '--use_appearance', '--batch_size', '1', '--pre_norm', '--image', image_path], check=True, capture_output=True, text=True)
        print(result.stdout)  # Print the output of the target script
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")
        QMessageBox.critical(None, "Error", f"An error occurred: {e.stderr}")

if __name__ == "__main__":
    main()
