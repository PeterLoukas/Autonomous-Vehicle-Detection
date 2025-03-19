"""
YOLOv8 Inference Script 🏎️🚦

📌 This script is a **standalone** inference pipeline that replicates the last cell 
   of the Google Colab notebook: `07_Inference_pytorch.ipynb`  
   
🔹 **Model:** Trained YOLOv8 (custom dataset)  
🔹 **Purpose:** Perform inference on test images/videos  
🔹 **Settings:** Exact settings from the last cell of Colab Notebook  
🔹 **Output Format:** YOLO labels + annotated images  

💡 This script is optimized for **direct execution** in local environments 
   (Windows/Linux/Mac) without needing to manually set up Colab.

🛠 **Key Features**
✅ Loads the fine-tuned YOLOv8 model (best.pt)
✅ Runs inference on images/videos with emergency and non-emergency vehicles 
✅ Keeps the same inference arguments such as, **IoU, confidence threshold, batch size, and augmentations**  
✅ Saves the results 'images' and 'labels' in a custom directory 

📍 **Colab Reference:**
This script **mirrors** the last cell of `07_Inference_pytorch.ipynb`:
```python
results = model.predict(
    source=source,
    imgsz=(1024, 1024),
    conf=0.43,
    batch=1,
    device=0,
    stream=True,
    line_width=1,
    iou=0.4,
    agnostic_nms=True,
    augment=True,
    save=True,
    save_txt=True,
    save_conf=True,
    show_labels=False,
    show_conf=False,
    project=save_dir
)
"""

import argparse
import os
from ultralytics import YOLO

def run_inference(model_path, source, output, imgsz, conf, iou, batch, device, 
                  max_det, vid_stride, stream, stream_buffer, visualize, augment, 
                  agnostic_nms, classes, retina_masks, embed, project, name, 
                  show, save, save_frames, save_txt, save_conf, save_crop, 
                  show_labels, show_conf, show_boxes, line_width, verbose, half):
    """Runs YOLOv8 inference on the given source and saves results."""
    
    # Validate model path
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Validate source directory or file
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source directory or file not found: {source}")

    # Create default output directory if not specified
    if output is None:
        output = os.path.join(os.getcwd(), "inference_results")

    os.makedirs(output, exist_ok=True)

    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Run inference with streaming mode enabled by default
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        batch=batch,
        device=device,
        max_det=max_det,
        vid_stride=vid_stride,
        stream=stream,  
        stream_buffer=stream_buffer,
        visualize=visualize,
        augment=augment,
        agnostic_nms=agnostic_nms,
        classes=classes,
        retina_masks=retina_masks,
        embed=embed,
        project=project,
        name=name,
        show=show,
        save=save,
        save_frames=save_frames,
        save_txt=save_txt,
        save_conf=save_conf,
        save_crop=save_crop,
        show_labels=show_labels,
        show_conf=show_conf,
        show_boxes=show_boxes,
        line_width=line_width,
        verbose=verbose,
        half=half
    )

    # Process streamed results and display logs
    for result in results:
        print(f"Processed: {os.path.basename(result.path)}")

def main():
    """Main function to handle command-line arguments and run inference."""
    
    parser = argparse.ArgumentParser(
        description="YOLOv8 Inference Script: Run object detection on images/videos.",
        epilog="Example usage:\npython inference.py --model-path best.pt --source dataset/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Define CLI arguments with defaults
    parser.add_argument("--model-path", type=str, required=True, help="Path to the YOLOv8 model file (best.pt).")
    parser.add_argument("--source", type=str, required=True, help="Path to the image/video file or directory for inference.")
    parser.add_argument("--output", type=str, default=None, help="Output directory for saving results. Defaults to 'inference_results'.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size for inference.")
    parser.add_argument("--conf", type=float, default=0.43, help="Confidence threshold for detections.")
    parser.add_argument("--iou", type=float, default=0.4, help="IoU threshold for Non-Maximum Suppression (NMS).")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for inference. Only applicable for directories and videos.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (e.g., 'cpu', 'cuda:0').")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum number of detections per image.")
    parser.add_argument("--vid-stride", type=int, default=1, help="Frame stride for video inputs.")
    parser.add_argument("--stream", type=bool, default=True, help="Enable streaming mode to prevent RAM overflow.")  
    parser.add_argument("--stream-buffer", type=bool, default=False, help="Queue incoming frames in a buffer for video streams.")
    parser.add_argument("--visualize", type=bool, default=False, help="Enable visualization of model features during inference.")
    parser.add_argument("--augment", type=bool, default=True, help="Enable test-time augmentation for improved robustness.")
    parser.add_argument("--agnostic-nms", type=bool, default=True, help="Enable class-agnostic NMS.")
    parser.add_argument("--classes", type=list, default=None, help="Filter predictions to specific class IDs.")
    parser.add_argument("--retina-masks", type=bool, default=False, help="Return high-resolution segmentation masks.")
    parser.add_argument("--embed", type=list, default=None, help="Extract feature vectors from specific layers.")
    parser.add_argument("--project", type=str, default="inference_results", help="Main directory for results.")
    parser.add_argument("--name", type=str, default="predictions", help="Name of the prediction run.")
    parser.add_argument("--show", type=bool, default=False, help="Display images/videos during inference.")
    parser.add_argument("--save", type=bool, default=True, help="Save annotated images/videos.")
    parser.add_argument("--save-frames", type=bool, default=False, help="Save individual video frames as images.")
    parser.add_argument("--save-txt", type=bool, default=True, help="Save detection results in text format.")
    parser.add_argument("--save-conf", type=bool, default=True, help="Include confidence scores in the saved text files.")
    parser.add_argument("--save-crop", type=bool, default=False, help="Save cropped images of detected objects.")
    parser.add_argument("--show-labels", type=bool, default=False, help="Show class labels in visualization.")
    parser.add_argument("--show-conf", type=bool, default=False, help="Show confidence scores in visualization.")
    parser.add_argument("--show-boxes", type=bool, default=True, help="Draw bounding boxes around detected objects.")
    parser.add_argument("--line-width", type=int, default=1, help="Line width of bounding boxes.")
    parser.add_argument("--verbose", type=bool, default=True, help="Display detailed inference logs.")
    parser.add_argument("--half", type=bool, default=False, help="Enable FP16 half-precision inference.")

    # Parse arguments
    args = parser.parse_args()

    try:
        run_inference(
            args.model_path,
            args.source,
            args.output,
            args.imgsz,
            args.conf,
            args.iou,
            args.batch,
            args.device,
            args.max_det,
            args.vid_stride,
            args.stream,
            args.stream_buffer,
            args.visualize,
            args.augment,
            args.agnostic_nms,
            args.classes,
            args.retina_masks,
            args.embed,
            args.project,
            args.name,
            args.show,
            args.save,
            args.save_frames,
            args.save_txt,
            args.save_conf,
            args.save_crop,
            args.show_labels,
            args.show_conf,
            args.show_boxes,
            args.line_width,
            args.verbose,
            args.half
        )
    except Exception as e:
        parser.error(str(e))

if __name__ == "__main__":
    main()