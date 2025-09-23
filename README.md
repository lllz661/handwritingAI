# Handwriting Recognition System

An end-to-end pipeline for handwritten text recognition using deep learning. The system processes PDF documents, detects text blocks, and recognizes handwritten text using a CRNN (Convolutional Recurrent Neural Network) model.

## Features

- **PDF Processing**: Convert PDF documents to images
- **Text Block Detection**: Automatically detect and crop text blocks from pages
- **Text Recognition**: Recognize handwritten text using a trained CRNN model
- **Training Pipeline**: Train custom models on your own dataset
- **Evaluation**: Comprehensive metrics including CER, WER, and accuracy

## Project Structure

```
handwritingAI/
├── config.yaml             # Configuration file
├── data/                   # Data directories
│   ├── raw/               # Raw input files (PDFs)
│   ├── processed/         # Processed data (images, crops)
│   └── ground_truth/      # Ground truth text files
├── models/                # Model checkpoints
│   ├── checkpoints/      # Training checkpoints
│   └── best/             # Best performing models
├── notebooks/            # Jupyter notebooks for exploration
├── outputs/              # Output files
│   ├── logs/            # Training logs
│   └── results/         # Evaluation results
└── src/                  # Source code
    ├── data/            # Data loading and processing
    ├── models/          # Model architectures
    ├── preprocessing/   # Data preprocessing
    ├── training/        # Training logic
    ├── evaluation/      # Evaluation metrics
    └── utils/           # Utility functions
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/handwritingAI.git
   cd handwritingAI
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Prepare Your Data

- Place your PDF files in `data/raw/pdf/`
- Add corresponding ground truth text files in `data/ground_truth/` with matching names

### 2. Run the Pipeline

#### Option 1: Run the complete pipeline
```bash
python runner.py run_all
```

#### Option 2: Run individual steps
```bash
# Preprocess data (PDF → images → text blocks)
python runner.py preprocess

# Train the model
python runner.py train

# Evaluate a trained model
python runner.py evaluate --model models/best/crnn.pth

# Make predictions on an image
python runner.py predict path/to/image.png
```

### 3. View Results

- Training logs: `outputs/logs/training.log`
- Evaluation results: `outputs/results/`
- Processed images: `data/processed/`

## Configuration

Edit `config.yaml` to customize the pipeline:

- **Data paths**: Update file paths as needed
- **Model parameters**: Adjust model architecture and training settings
- **Preprocessing**: Configure image processing options

## Training on Custom Data

1. Prepare your dataset in the following structure:
   ```
   data/
   ├── raw/pdf/
   │   ├── document1.pdf
   │   └── document2.pdf
   └── ground_truth/
       ├── document1.txt
       └── document2.txt
   ```

2. Update `config.yaml` with your desired settings

3. Run the training pipeline:
   ```bash
   python runner.py train
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [OpenCV](https://opencv.org/) - Image processing
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF processing
- [CTC Loss](https://distill.pub/2017/ctc/) - Connectionist Temporal Classification
