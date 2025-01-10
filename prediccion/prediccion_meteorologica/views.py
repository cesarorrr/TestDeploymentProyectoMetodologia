from django.shortcuts import render
import joblib
import datetime

def predict_weather(request):
    prediction = None
    weather_info = None

    if request.method == 'POST':
        data = request.POST

        # 1. Extraer la fecha del formulario
        date_str = data.get('date')  # Asegúrate que en el formulario el input tenga name="date"
        date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')

        # 2. Extraer year, month, day
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day

        # 3. Construir el vector de características
        #    *Nota*: Esto debe concordar con el orden de columnas que usaste al entrenar el modelo
        features = [
            float(data['precipitation']),
            float(data['temp_max']),
            float(data['temp_min']),
            float(data['wind']),
            float(data['humidity']),
            float(data['pressure']),
            float(data['solar_radiation']),
            float(data['visibility']),
            year,
            month,
            day
        ]

        # 4. Cargar modelo y scaler
        model = joblib.load("prediccion_meteorologica/models/svm_model.pkl")
        scaler = joblib.load("prediccion_meteorologica/models/scaler.pkl")

        # 5. Escalar las características y predecir
        features_scaled = scaler.transform([features])
        prediction = int(model.predict(features_scaled)[0])

        # 6. Diccionario para obtener la descripción del clima según la predicción
        weather_dict = {
            1: {
                'name': "Tormenta",
                'description': "Se esperan fuertes tormentas. ¡Precaución!",
                'image': "https://media.giphy.com/media/6ZhkSxi5KvORq/giphy.gif",
                'icon': "⚡"
            },
            2: {
                'name': "Lluvia",
                'description': "Día lluvioso, no olvides tu paraguas.",
                'image': "https://media.giphy.com/media/Ckt7qu9ksg5ByO2ibi/giphy.gif",
                'icon': "🌧"
            },
            3: {
                'name': "Nublado",
                'description': "Cielo cubierto y con pocas probabilidades de sol.",
                'image': "https://media.giphy.com/media/RO5XhlFWOPs6k/giphy.gif",
                'icon': "☁"
            },
            4: {
                'name': "Niebla",
                'description': "Visibilidad reducida por la niebla.",
                'image': "https://media.giphy.com/media/W0sgn9xy8Mul3ab0mG/giphy.gif",
                'icon': "🌫"
            },
            5: {
                'name': "Soleado",
                'description': "Día despejado y radiante. ¡A disfrutar!",
                'image': "https://media.giphy.com/media/bcJvDLgxVSPulkDYB4/giphy.gif",
                'icon': "☀"
            }
        }

        weather_info = weather_dict.get(prediction, None)

    return render(request, 'index.html', {
        'prediction': prediction,
        'weather_info': weather_info
    })
