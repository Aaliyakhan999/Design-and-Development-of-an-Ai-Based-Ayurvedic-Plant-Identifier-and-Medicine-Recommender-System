from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
import base64
import os
import numpy as np
from ultralytics import YOLO
from PIL import Image
from .models import ChatSession, ChatMessage
import uuid
from django.shortcuts import render
from google.generativeai import configure, GenerativeModel
import urllib.request
import json
import google.generativeai as genai


# Configure Gemini API
configure(api_key="Your API Key")
model = GenerativeModel('gemini-1.5-flash')

# Define Ayurvedic expert prompt
AYURVEDIC_CONTEXT = (
    "You are an expert in Ayurvedic medicine and herbal remedies. "
    "You only answer questions related to Ayurveda, medicinal plants, natural healing methods, "
    "home remedies, and plant-based treatments. If the question is not related to Ayurveda, "
    "politely inform the user that you can only help with Ayurvedic topics."
)

def chat_view(request):
    # Get or create a session ID
    if 'chat_session_id' not in request.session:
        request.session['chat_session_id'] = str(uuid.uuid4())
        request.session.save()

    session_id = request.session['chat_session_id']
    
    # Get or create chat session in the database
    chat_session, created = ChatSession.objects.get_or_create(session_id=session_id)
    
    if request.method == "POST":
        question = request.POST.get("question")
        
        if question:
            # Save user message
            user_message = ChatMessage.objects.create(
                chat_session=chat_session,
                is_user=True,
                content=question
            )
            
            try:
                # Add context to the user question
                prompt = f"{AYURVEDIC_CONTEXT}\n\nUser: {question}\nAyurvedic Expert:"
                
                # Generate response
                response = model.generate_content(prompt)
                answer = response.text
                
                # Save bot response
                bot_message = ChatMessage.objects.create(
                    chat_session=chat_session,
                    is_user=False,
                    content=answer
                )
            except Exception as e:
                print(f"Error: {e}")
                bot_message = ChatMessage.objects.create(
                    chat_session=chat_session,
                    is_user=False,
                    content="Sorry, I couldn't process that request."
                )
    
    # Get all messages for this chat session
    messages = ChatMessage.objects.filter(chat_session=chat_session).order_by('timestamp')
    
    # Format messages for the template
    chat_history = []
    current_user_message = None
    
    for message in messages:
        if message.is_user:
            current_user_message = message.content
        else:
            if current_user_message:
                chat_history.append({
                    'user': current_user_message,
                    'bot': message.content
                })
                current_user_message = None
    
    if current_user_message:
        chat_history.append({
            'user': current_user_message,
            'bot': "Processing..."
        })
    
    return render(request, 'chatbot.html', {
        'chat_history': chat_history
    })



def recipe_view(request):
    context = {
        'last_disease': '',
        'last_recipe': '',
    }
    
    if request.method == 'POST':
        disease = request.POST.get('question', '').strip()
        if disease:
            # Save the disease to context
            context['last_disease'] = disease
            
            # Get recipe from Gemini API
            recipe = get_ayurvedic_recipe(disease)
            
            # Save the recipe to context
            context['last_recipe'] = recipe
    
    return render(request, 'recipe.html', context)

def get_ayurvedic_recipe(disease):
    # List of common diseases/conditions that can be treated with Ayurvedic remedies
    valid_diseases = [
        "diabetes", "arthritis", "asthma", "hypertension", "obesity", "insomnia", 
        "anxiety", "depression", "digestive disorders", "acidity", "constipation", 
        "diarrhea", "piles", "skin disorders", "eczema", "psoriasis", "acne", 
        "respiratory infections", "common cold", "cough", "fever", "headache", 
        "migraine", "allergies", "sinusitis", "thyroid disorders", "hair loss", 
        "menstrual disorders", "urinary tract infections", "kidney stones", 
        "liver disorders", "jaundice", "anemia", "fatigue", "stress", "joint pain",
        "back pain", "rheumatism", "gastritis", "ulcers", "irritable bowel syndrome",
        "cholesterol", "heart disease", "alzheimer's", "parkinson's", "osteoporosis"
    ]
    
    # Convert to lowercase and check if the input is a valid disease
    disease_lower = disease.lower()
    
    # Check if the input resembles a valid disease
    is_valid = False
    matched_disease = ""
    
    for valid_disease in valid_diseases:
        if disease_lower == valid_disease or disease_lower in valid_disease or valid_disease in disease_lower:
            is_valid = True
            matched_disease = valid_disease
            break
    
    if not is_valid:
        return "Sorry, I can only provide Ayurvedic remedies for recognized health conditions. Please enter a valid disease or health condition."
    
    try:
        # Configure the Gemini API
        genai.configure(api_key="Your API Key")
        
        # Initialize the model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Create a prompt that specifically asks for Ayurvedic recipes with plant/flower ingredients
        prompt = f"""Create an Ayurvedic recipe for treating {matched_disease or disease}. The recipe must include plant or flower ingredients. Format the recipe with:
1. A brief title
2. List of ingredients (focus on herbs, plants, and flowers)
3. Preparation method
4. Dosage instructions
5. Brief explanation of how these ingredients help with {matched_disease or disease}
Only provide the recipe without any other information or disclaimers."""
        
        # Generate content using the model
        response = model.generate_content(prompt)
        
        # Extract the text from the response
        recipe_text = response.text
        return recipe_text
            
    except Exception as e:
        return f"Error processing recipe request: {str(e)}"
    
# Define class labels for YOLOv8 model
LEAF_CLASSES = [
    'AloeVera', 'Amaltas', 'Amla', 'Anantamul', 'Aquatic-Ginger', 'Ashoka',
    'Ashwagandha', 'Belladonna', 'Cardamom', 'Castor Oil Plant', 'Celosia-Argentea',
    'Chinese-Ixora', 'Chirata', 'Coleus', 'Dill', 'Gokharu', 'Gugal',
    'Indian Pulai', 'Ipecac', 'Isabghul', 'Kutaja', 'Lipstick-Plant',
    'Malabar Nut', 'Medicinal-Arive Dantu', 'Medicinal-Basale', 'Medicinal-Neem',
    'Medicinal-Rose Apple', 'Medicinal-Sandalwood', 'Meetha', 'OLEANDER',
    'Oleander', 'Pine', 'Sandalwood', 'Simpoh-Lak', 'Somavalli', 'Sweet Basil',
    'Tamarind', 'Thyme', 'Ti', 'Yam', 'ambrosia', 'lemongrass', 'neem', 'tulsi'
]

# Load YOLOv8 model (lazy loading)
yolo_model = None

def load_model():
    global yolo_model
    if yolo_model is None:
        try:
            # Try to load the model from the root directory
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'medicinal_leaf_detector.pt')
            if os.path.exists(model_path):
                yolo_model = YOLO(model_path)
                print(f"Successfully loaded YOLOv8 model from {model_path}")
            else:
                print(f"Model file not found at {model_path}")
                raise FileNotFoundError(f"Model file not found at {model_path}")
        except Exception as e:
            print(f"Error loading YOLOv8 model: {str(e)}")
            raise

# Home Page
def home(request):
    return render(request, 'home.html')

# About Page
def about(request):
    return render(request, 'about.html')

# Contact Page
def contact(request):
    return render(request, 'contact.html')

# Login View
def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = authenticate(username=email, password=password)
        if user is not None:
            login(request, user)
            return redirect('profile')
        else:
            messages.error(request, 'Invalid email or password')
    return render(request, 'login.html')

# Signup View
def signup(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm-password')
        
        if password != confirm_password:
            messages.error(request, 'Passwords do not match')
            return render(request, 'signup.html', {
                'name': name,
                'email': email
            })
        
        if User.objects.filter(username=email).exists():
            messages.error(request, 'An account with this email already exists.')
            return render(request, 'signup.html', {
                'name': name,
                'email': email
            })
        
        # Create user — let the signal handle Profile creation
        user = User.objects.create_user(username=email, email=email, password=password)
        user.first_name = name
        user.save()

        messages.success(request, 'Account created successfully! Please login.')
        return redirect('login')
    
    return render(request, 'signup.html')

# Profile View
@login_required
def profile(request):
    return render(request, 'profile.html', {'user': request.user})

# Sample medicinal information for common plants
PLANT_MEDICINAL_INFO = {
   'AloeVera': 'Aloe vera is used for treating skin conditions, burns, and digestive issues. It has anti-inflammatory and antibacterial properties. The gel from its leaves can be applied topically for sunburns, minor cuts, and skin irritations.',
    
    'Amla': 'Amla (Indian Gooseberry) is rich in vitamin C and antioxidants. Used for improving immunity, digestion, and hair health. It supports liver function and is used in traditional Ayurvedic medicine for various ailments.',
    
    'Tulsi': 'Tulsi (Holy Basil) is used for respiratory disorders, stress, and anxiety. It has antimicrobial and anti-inflammatory properties. The leaves can be consumed as tea to boost immunity and relieve cold symptoms.',
    
    'Neem': 'Neem has antibacterial, antifungal, and antiviral properties. Used for skin disorders, diabetes management, and dental issues. Its leaves, bark, and oil are all utilized in traditional medicine.',
    
    'Tamarind': 'Tamarind is rich in antioxidants and has anti-inflammatory properties. It is used to treat digestive issues, fever, and sore throat. The pulp is a good source of iron and can help with anemia.',
    
    'Lemongrass': 'Lemongrass has antifungal, antibacterial, and anti-inflammatory properties. It helps with digestive issues, reduces anxiety, and may help lower cholesterol. Used as tea or in cooking.',
    
    'Amaltas': 'Amaltas (Golden Shower Tree) is known for its laxative properties. It is used to treat constipation, fever, skin diseases, and liver disorders in traditional medicine.',
    
    'Anantamul': 'Anantamul (Indian Sarsaparilla) is used for detoxification, improving skin complexion, and treating urinary infections. It has anti-inflammatory and blood-purifying properties.',
    
    'Aquatic-Ginger': 'Aquatic ginger has anti-inflammatory and digestive properties. It is used in traditional medicine for gastrointestinal relief and improving appetite.',
    
    'Ashoka': 'Ashoka tree bark is widely used in Ayurveda to treat gynecological disorders such as menstrual problems and uterine issues. It also has anti-inflammatory properties.',
    
    'Ashwagandha': 'Ashwagandha is an adaptogenic herb known for reducing stress, enhancing stamina, and improving overall vitality. It supports immune and endocrine systems.',
    
    'Belladonna': 'Belladonna (Deadly Nightshade) is used in homeopathy to treat pain, inflammation, and cold symptoms, but it is highly toxic in large doses and should be used with caution.',
    
    'Cardamom': 'Cardamom aids digestion, reduces nausea, and helps with respiratory issues. It also has antioxidant and diuretic properties that support heart health.',
    
    'Castor Oil Plant': 'Castor oil is used as a laxative and for treating skin disorders. Its leaves may be applied to reduce inflammation and relieve pain in traditional medicine.',
    
    'Celosia-Argentea': 'Celosia is used in folk medicine to treat diarrhea, wounds, and inflammation. It is considered a cooling herb and supports digestive and immune health.',
    
    'Chinese-Ixora': 'Chinese Ixora is used for treating dysentery, diarrhea, and wounds. Its flowers and leaves are used in decoctions for their anti-inflammatory and astringent properties.',
    
    'Chirata': 'Chirata is a bitter herb used in Ayurveda to treat fever, malaria, and liver disorders. It has anti-inflammatory, antioxidant, and antidiabetic properties.',
    
    'Coleus': 'Coleus (Patharchur) is used for respiratory issues, digestion, and skin conditions. It may support weight management and reduce blood pressure.',
    
    'Dill': 'Dill is used to relieve digestive disorders, flatulence, and colic in infants. It has antimicrobial and antispasmodic properties.',
    
    'Gokharu': 'Gokharu (Tribulus terrestris) is used to improve kidney function, treat urinary tract issues, and enhance vitality. Often used for male reproductive health.',
    
    'Gugal': 'Gugal (Commiphora mukul) is used in Ayurveda to manage arthritis, obesity, and high cholesterol. It has anti-inflammatory and lipid-lowering effects.',
    
    'Indian Pulai': 'Indian Pulai (Alstonia scholaris) bark is used to treat respiratory disorders like asthma and bronchitis, and also for malarial fevers and skin diseases.',
    
    'Ipecac': 'Ipecac is used to induce vomiting in poisoning cases and treat bronchitis. It has expectorant properties but must be used with caution due to its potency.',
    
    'Isabghul': 'Isabghul (Psyllium husk) is a natural fiber used to treat constipation and support digestive health. It may also help regulate cholesterol and blood sugar levels.',
    
    'Kutaja': 'Kutaja (Holarrhena antidysenterica) is used to treat diarrhea, dysentery, and other gastrointestinal issues. It has antimicrobial and antidiarrheal properties.',
    
    'Lipstick-Plant': 'The lipstick plant has limited traditional use but may contain compounds with antimicrobial and anti-inflammatory effects. More research is needed for medicinal validation.',
    
    'Malabar Nut': 'Malabar Nut (Adhatoda vasica) is used to treat cough, bronchitis, and asthma. It has expectorant and bronchodilator properties and is a key herb in respiratory care.',
    
    'Medicinal-Arive Dantu': 'Arive Dantu is used in traditional medicine to treat wounds and skin infections. Its leaves are believed to have antiseptic and healing properties.',
    
    'Medicinal-Basale': 'Basale (Malabar spinach) has cooling and laxative properties. Used in Ayurveda for ulcers, inflammation, and digestive health.',
    
    'Medicinal-Neem': 'Neem (duplicate normalized) – see Neem above.',
    
    'Medicinal-Rose Apple': 'Rose apple is used to treat diabetes and digestive disorders. The leaves and seeds may have antimicrobial and antihyperglycemic effects.',
    
    'Medicinal-Sandalwood': 'Sandalwood is used in Ayurveda for skin care, treating inflammation, and calming the mind. It has antiseptic and cooling properties.',
    
    'Meetha': 'Meetha (Sweet Leaf or Stevia) is used as a natural sweetener for diabetic care. It may help regulate blood sugar levels and has antioxidant properties.',
    
    'OLEANDER': 'Oleander is a highly toxic plant, though it has been used in small doses in homeopathy for heart conditions and skin issues. Caution is critical.',
    
    'Oleander': 'Same as OLEANDER — duplicate name with different casing.',
    
    'Pine': 'Pine leaves and bark extracts are used to treat respiratory issues, reduce inflammation, and promote wound healing. Pine oil is also used for its antimicrobial effects.',
    
    'Sandalwood': 'See Medicinal-Sandalwood — has calming, anti-inflammatory, and antimicrobial properties used for skin and aromatherapy.',
    
    'Simpoh-Lak': 'Simpoh Lak (Dillenia suffruticosa) is used in Southeast Asian traditional medicine to treat wounds, diarrhea, and as an anti-inflammatory agent.',
    
    'Somavalli': 'Somavalli is believed to help with immunity and digestion in traditional practices, although scientific evidence is limited.',
    
    'Sweet Basil': 'Sweet Basil is rich in antioxidants and essential oils. Used to reduce inflammation, manage stress, and treat respiratory and digestive issues.',
    
    'Thyme': 'Thyme has antiseptic and antibacterial properties. It is used to treat respiratory infections, sore throat, and digestive issues.',
    
    'Ti': 'Ti plant leaves are used externally to treat wounds, fever, and skin conditions. Decoctions are sometimes used in traditional herbal practices.',
    
    'Yam': 'Yam is rich in fiber, potassium, manganese, and antioxidants. It supports brain function, reduces inflammation, and helps manage blood sugar levels. Used in traditional medicine for menopause and digestion.',
    
    'ambrosia': 'Ambrosia (Ragweed) is used traditionally to treat fever, colds, and nausea. It also has anti-inflammatory properties but is a common allergen and should be used with caution.',
}

# Add extra entries for variants and to ensure broader coverage
for i, plant in enumerate(LEAF_CLASSES):
    if plant not in PLANT_MEDICINAL_INFO:
        PLANT_MEDICINAL_INFO[plant] = f"{plant} is used in traditional medicine for its therapeutic properties. It may have anti-inflammatory, antioxidant, or antimicrobial benefits depending on the specific species."

# Helper function to get medicinal info for a plant
def get_medicinal_info(plant_name):
    # Remove "_Plant" suffix if present and clean the name
    clean_name = plant_name.replace('_Plant', '').strip()
    
    # Try exact match first
    if clean_name in PLANT_MEDICINAL_INFO:
        return PLANT_MEDICINAL_INFO[clean_name]
    
    # If no exact match, find partial matches
    partial_matches = []
    for key in PLANT_MEDICINAL_INFO:
        if key.lower() in clean_name.lower() or clean_name.lower() in key.lower():
            partial_matches.append(key)
    
    if partial_matches:
        # Use the first partial match
        match = partial_matches[0]
        return PLANT_MEDICINAL_INFO[match]
    
    # If still no match, return default text
    return f"{clean_name} is a medicinal plant used in traditional medicine. Please consult an herbal specialist for specific medicinal properties."

# Prediction View with YOLOv8
@login_required
def prediction(request):
    prediction_results = {}
    
    if request.method == 'POST':
        if request.FILES.get('plant_image'):
            image_file = request.FILES['plant_image']
            fs = FileSystemStorage()
            image_path = fs.save(image_file.name, image_file)
            image_full_path = fs.path(image_path)

            try:
                # Load model
                load_model()
                
                # Load and preprocess the image
                img = Image.open(image_full_path)
                
                # Make predictions using YOLOv8
                results = yolo_model(img)
                
                # Process results
                if len(results) > 0:
                    result = results[0]  # Get first image result
                    
                    # Check if we have classification results
                    if hasattr(result, 'names') and hasattr(result, 'boxes'):
                        # Get the detected objects
                        boxes = result.boxes
                        
                        if len(boxes) > 0:
                            # Get the class with highest confidence
                            confidences = boxes.conf.cpu().numpy()
                            class_ids = boxes.cls.cpu().numpy().astype(int)
                            
                            # Get top 3 predictions
                            top3_indices = np.argsort(confidences)[-3:][::-1]
                            top3_classes = [(result.names[class_ids[i]], float(confidences[i]) * 100) 
                                          for i in top3_indices]
                            
                            best_idx = top3_indices[0]
                            best_class = result.names[class_ids[best_idx]]
                            best_confidence = float(confidences[best_idx]) * 100
                            
                            prediction_results['leaf'] = {
                                'class': best_class,
                                'confidence': best_confidence,
                                'top3': top3_classes
                            }
                        else:
                            # No objects detected
                            messages.warning(request, "No medicinal plants detected in the image. Please try another image.")
                    else:
                        # Try to get classification results if available
                        if hasattr(result, 'probs') and result.probs is not None:
                            # Get top 3 predictions
                            probs = result.probs.data.cpu().numpy()
                            top3_indices = np.argsort(probs)[-3:][::-1]
                            top3_classes = [(LEAF_CLASSES[i], float(probs[i]) * 100) 
                                          for i in top3_indices]
                            
                            best_idx = top3_indices[0]
                            best_class = LEAF_CLASSES[best_idx]
                            best_confidence = float(probs[best_idx]) * 100
                            
                            prediction_results['leaf'] = {
                                'class': best_class,
                                'confidence': best_confidence,
                                'top3': top3_classes
                            }
                        else:
                            # No classification results
                            messages.warning(request, "Could not classify the image. Please try another image.")
                
                # Encode image for display
                with open(image_full_path, "rb") as img_file:
                    base64_image = base64.b64encode(img_file.read()).decode("utf-8")
                prediction_results['image'] = base64_image
                
                # Get medicinal information
                if 'leaf' in prediction_results:
                    plant_name = prediction_results['leaf']['class']
                    prediction_results['details'] = get_medicinal_info(plant_name)
                
            except Exception as e:
                messages.error(request, f"Error processing image: {str(e)}")
                import traceback
                print(f"Error in prediction view: {str(e)}")
                print(traceback.format_exc())
                
    return render(request, 'prediction.html', {
        'prediction_results': prediction_results
    })

# Team Page
def team(request):
    return render(request, 'team.html')

@login_required
def upload_profile_image(request):
    if request.method == 'POST' and request.FILES.get('profile_image'):
        profile = request.user.profile
        profile.image = request.FILES['profile_image']
        profile.save()
        return JsonResponse({'success': True})
    return JsonResponse({'success': False, 'error': 'No image provided'})
