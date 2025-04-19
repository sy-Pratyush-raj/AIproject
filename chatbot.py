from flask import Flask, request, jsonify
import numpy as np
from scipy.io import wavfile
import io
import soundfile as sf
import librosa
import tempfile
import os
from datetime import datetime

app = Flask(__name__)

# Mock user database for demonstration
users = {
    "demo": {
        "stress_history": [],
        "baseline_features": None
    }
}

# Stress advice knowledge base
STRESS_ADVICE = {
    "happy": [
        "Practice gratitude daily by writing down 3 things you're thankful for",
        "Engage in activities you enjoy, even for just 15 minutes a day",
        "Connect with loved ones - social connections boost happiness",
        "Exercise regularly - even a short walk can improve your mood",
        "Help others - acts of kindness increase personal happiness"
    ],
    "stress": [
        "Try the 4-7-8 breathing technique: inhale for 4s, hold for 7s, exhale for 8s",
        "Take a 5-minute break to stretch or walk around",
        "Practice mindfulness meditation for just 5 minutes",
        "Write down what's stressing you - getting it out of your head helps",
        "Listen to calming music or nature sounds"
    ],
    "anxious": [
        "Ground yourself with the 5-4-3-2-1 technique: name 5 things you see, 4 you can touch, etc.",
        "Try progressive muscle relaxation - tense and release each muscle group",
        "Limit caffeine and sugar which can worsen anxiety",
        "Challenge anxious thoughts by asking 'Is this really likely to happen?'",
        "Create a 'worry period' - postpone anxious thoughts to a specific time later"
    ],
    "angry": [
        "Count slowly to 10 before reacting",
        "Use 'I' statements instead of blame ('I feel...' rather than 'You always...')",
        "Channel energy into physical activity like brisk walking",
        "Practice deep breathing - inhale deeply through nose, exhale slowly through mouth",
        "Remove yourself from the situation temporarily if possible"
    ]
}

def extract_audio_features(audio_data, sr=16000):
    """Extract audio features that correlate with stress"""
    try:
        # Convert bytes to numpy array
        y, sr = librosa.load(io.BytesIO(audio_data), sr=sr)
        
        # Extract features
        features = {}
        
        # Pitch (fundamental frequency)
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0 = f0[f0 > 0]  # Remove zeros
        if len(f0) > 0:
            features['pitch_mean'] = np.mean(f0)
            features['pitch_std'] = np.std(f0)
            features['pitch_range'] = np.max(f0) - np.min(f0)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_range'] = 0
        
        # Speaking rate (approximate)
        rms = librosa.feature.rms(y=y)
        voiced_frames = np.sum(rms > np.median(rms))
        features['speaking_rate'] = voiced_frames / (len(y) / sr)
        
        # Jitter (pitch variability)
        if len(f0) > 1:
            features['jitter'] = np.mean(np.abs(np.diff(f0)))
        else:
            features['jitter'] = 0
        
        # Shimmer (amplitude variability)
        if len(f0) > 1:
            amplitudes = librosa.amplitude_to_db(librosa.feature.rms(y=y))
            features['shimmer'] = np.mean(np.abs(np.diff(amplitudes)))
        else:
            features['shimmer'] = 0
            
        # MFCCs (voice quality)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1).tolist()
        
        # Harmonic-to-noise ratio
        features['hnr'] = librosa.effects.harmonic(y).mean()
        
        return features
        
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def predict_stress_level(features, baseline=None):
    """Predict stress level based on audio features"""
    if baseline:
        pitch_dev = abs(features['pitch_mean'] - baseline['pitch_mean']) / baseline['pitch_mean']
        pitch_var_dev = abs(features['pitch_std'] - baseline['pitch_std']) / baseline['pitch_std']
        rate_dev = abs(features['speaking_rate'] - baseline['speaking_rate']) / baseline['speaking_rate']
        jitter_dev = abs(features['jitter'] - baseline['jitter']) / baseline['jitter']
        shimmer_dev = abs(features['shimmer'] - baseline['shimmer']) / baseline['shimmer']
    else:
        pitch_dev = max(0, (features['pitch_mean'] - 180) / 180)
        pitch_var_dev = max(0, (features['pitch_std'] - 30) / 30)
        rate_dev = max(0, (features['speaking_rate'] - 3) / 3)
        jitter_dev = max(0, (features['jitter'] - 1.5) / 1.5)
        shimmer_dev = max(0, (features['shimmer'] - 3) / 3)
    
    stress_score = (
        0.3 * pitch_dev + 
        0.2 * pitch_var_dev + 
        0.2 * rate_dev + 
        0.15 * jitter_dev + 
        0.15 * shimmer_dev
    )
    
    stress_level = min(100, max(0, stress_score * 80))
    return round(stress_level)

def get_stress_advice(keyword):
    """Get personalized advice based on stress-related keywords"""
    keyword = keyword.lower()
    
    # Find the best matching category
    if "happy" in keyword or "joy" in keyword or "happiness" in keyword:
        category = "happy"
    elif "stress" in keyword or "overwhelm" in keyword:
        category = "stress"
    elif "anxious" in keyword or "anxiety" in keyword or "worry" in keyword:
        category = "anxious"
    elif "angry" in keyword or "anger" in keyword or "frustrat" in keyword:
        category = "angry"
    else:
        category = "stress"  # Default
    
    advice_list = STRESS_ADVICE.get(category, STRESS_ADVICE["stress"])
    selected_advice = np.random.choice(advice_list, size=min(3, len(advice_list)), replace=False)
    
    return {
        "category": category,
        "advice": selected_advice,
        "message": f"Here are some suggestions that might help with {category}:"
    }

@app.route('/analyze_voice', methods=['POST'])
def analyze_voice():
    """Endpoint for voice stress analysis"""
    try:
        user_id = request.form.get('user_id', 'demo')
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
            
        audio_file = request.files['audio']
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            audio_file.save(tmp.name)
            with open(tmp.name, 'rb') as f:
                audio_data = f.read()
            os.unlink(tmp.name)
        
        features = extract_audio_features(audio_data)
        if not features:
            return jsonify({'error': 'Could not process audio'}), 400
        
        baseline = users.get(user_id, {}).get('baseline_features')
        stress_level = predict_stress_level(features, baseline)
        
        if user_id not in users:
            users[user_id] = {'stress_history': [], 'baseline_features': None}
        
        users[user_id]['stress_history'].append({
            'timestamp': datetime.now().isoformat(),
            'stress_level': stress_level,
            'features': features
        })
        
        if baseline is None and len(users[user_id]['stress_history']) >= 3:
            avg_features = {
                'pitch_mean': np.mean([h['features']['pitch_mean'] for h in users[user_id]['stress_history'][:3]]),
                'pitch_std': np.mean([h['features']['pitch_std'] for h in users[user_id]['stress_history'][:3]]),
                'speaking_rate': np.mean([h['features']['speaking_rate'] for h in users[user_id]['stress_history'][:3]]),
                'jitter': np.mean([h['features']['jitter'] for h in users[user_id]['stress_history'][:3]]),
                'shimmer': np.mean([h['features']['shimmer'] for h in users[user_id]['stress_history'][:3]])
            }
            users[user_id]['baseline_features'] = avg_features
        
        if stress_level < 30:
            feedback = "Your voice shows low stress levels. You sound calm and relaxed."
        elif stress_level < 70:
            feedback = "Your voice shows moderate stress levels. Consider some relaxation techniques."
        else:
            feedback = "Your voice shows high stress levels. Try deep breathing or taking a short break."
        
        return jsonify({
            'stress_level': stress_level,
            'feedback': feedback,
            'features': features
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint for text chat interaction with stress advice"""
    try:
        message = request.json.get('message', '')
        user_id = request.json.get('user_id', 'demo')
        
        # Check for stress-related keywords
        stress_keywords = [
            'stress', 'anxious', 'anxiety', 'worry', 'worried',
            'happy', 'happiness', 'joy', 'angry', 'anger',
            'calm', 'relax', 'frustrat', 'overwhelm', 'depress',
            'how to', 'what should', 'advice', 'help me'
        ]
        
        msg_lower = message.lower()
        
        # If message contains stress-related keywords
        if any(keyword in msg_lower for keyword in stress_keywords):
            advice = get_stress_advice(msg_lower)
            return jsonify({
                'response': f"{advice['message']}\n\n1. {advice['advice'][0]}\n2. {advice['advice'][1]}\n3. {advice['advice'][2]}",
                'suggest_voice': False,
                'is_advice': True
            })
        
        # Default chatbot responses
        responses = {
            'hi': "Hello! I'm your AI stress tracker. Would you like me to analyze your voice for stress?",
            'hello': "Hi there! Click the microphone to let me analyze your stress levels.",
            'stress': "I can detect stress through your voice tone. Try speaking to me for analysis.",
            'help': "I analyze stress through voice patterns. Click the microphone and speak naturally for 5 seconds.",
            'default': "For stress analysis, please use the voice recording feature. Click the microphone icon."
        }
        
        response = responses['default']
        for keyword in responses:
            if keyword in msg_lower and keyword != 'default':
                response = responses[keyword]
                break
        
        return jsonify({
            'response': response,
            'suggest_voice': True,
            'is_advice': False
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return "AI Stress Level Tracker Backend is running! Use the frontend HTML file to interact."

if __name__ == '__main__':
    app.run(debug=True)