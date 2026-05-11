import google.generativeai as genai

genai.configure(api_key="AIzaSyATb_BFstWDrhDyRP6OCFEW9Pnn5JLWvjs")

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(model.name)