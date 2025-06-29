# Plant Medicine Identification System

This project uses YOLOv8 to classify medicinal plants and their leaves, providing information about their medicinal properties.

## Features

- Plant leaf image classification using YOLOv8
- Displays confidence scores for classifications
- Provides medicinal properties information for identified plants

## Setup

1. Clone this repository
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Make sure the YOLOv8 model file is in the correct location:
   - `medicinal_leaf_detector.pt` - for leaf classification (place in the root directory of the project)

4. Run the application:
   ```
   python manage.py runserver
   ```

5. Access the application at http://127.0.0.1:8000/

## Usage

1. Navigate to the Prediction page
2. Upload an image of a medicinal plant leaf
3. Click "Analyze" to get results

## Model

This system uses YOLOv8 for classification:
- Leaf classification model: Identifies 44 different medicinal plant leaves
- The model was trained on a dataset of medicinal plant leaves
- Classes include: AloeVera, Amaltas, Amla, Anantamul, Aquatic-Ginger, Ashoka, Ashwagandha, Belladonna, Cardamom, Castor Oil Plant, Celosia-Argentea, Chinese-Ixora, Chirata, Coleus, Dill, Gokharu, Gugal, Indian Pulai, Ipecac, Isabghul, Kutaja, Lipstick-Plant, Malabar Nut, Medicinal-Arive Dantu, Medicinal-Basale, Medicinal-Neem, Medicinal-Rose Apple, Medicinal-Sandalwood, Meetha, OLEANDER, Oleander, Pine, Sandalwood, Simpoh-Lak, Somavalli, Sweet Basil, Tamarind, Thyme, Ti, Yam, ambrosia, lemongrass, neem, tulsi

## Optional: Ollama Integration

The system can optionally use Ollama to provide detailed medicinal information. If you have Ollama installed:

1. Install Ollama following instructions at: https://ollama.com/
2. Run: `ollama run llama3.2-vision:latest`

This will enhance the plant information displayed with medicinal properties and uses. 