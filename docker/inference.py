import joblib
import pandas as pd
import warnings
import sklearn
print(sklearn.__version__)
warnings.simplefilter('ignore')
from sklearn.preprocessing import FunctionTransformer

import logging
from sys import stdout

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s %(levelname)s %(filename)s: %(message)s")
consoleHandler = logging.StreamHandler(stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)




# Definir el transformador personalizado para la fecha
def extract_month(X):
    """Extrae el mes de una columna de fechas."""
    
    # X llega como un DataFrame de (n_samples, 1)
    # Seleccionamos esa única columna con .iloc[:, 0] para obtener una pd.Series
    X_col = X.iloc[:, 0]

    # Convertir a datetime. 'coerce' convierte errores (o NaNs) en NaT.
    dates = pd.to_datetime(X_col, errors='coerce')
    
    # Extraer el mes. NaT (Not a Time) se convertirá en NaN.
    months = dates.dt.month
    
    # Devolverlo como (n_samples, 1) para sklearn
    return months.values.reshape(-1, 1)

def get_month_feature_names(transformer, input_features):
    return ['Month']

# Definir el transformador personalizado para la Amplitud Térmica
def calculate_thermal_amplitude(X):
    """Calcula la Amplitud Térmica: MaxTemp - MinTemp."""
    
    # 1. Asegura que sean numéricos (como ya lo tenías)
    # X viene como un array (N, 2) con [MinTemp, MaxTemp]
    max_temp = pd.to_numeric(X[:, 1], errors='coerce') 
    min_temp = pd.to_numeric(X[:, 0], errors='coerce')

    # 2. Calcula la amplitud térmica (el resultado es un numpy array)
    thermal_amplitude = (max_temp - min_temp)
    
    # 3. Haz reshape DIRECTAMENTE sobre el array de NumPy para que 
    # tenga la forma (n_samples, 1) esperada por el ColumnTransformer.
    return thermal_amplitude.reshape(-1, 1)

def get_amplitude_feature_names(transformer, input_features):
    return ['AmplitudTermica']

mapping_dict = {'Albury': 'Sur',
 'BadgerysCreek': 'Este',
 'Cobar': 'Este',
 'CoffsHarbour': 'Este',
 'Moree': 'Este',
 'Newcastle': 'Este',
 'NorahHead': 'Este',
 'NorfolkIsland': 'Este',
 'Penrith': 'Este',
 'Richmond': 'Sur',
 'Sydney': 'Este',
 'SydneyAirport': 'Este',
 'WaggaWagga': 'Sur',
 'Williamtown': 'Este',
 'Wollongong': 'Este',
 'Canberra': 'Sur',
 'Tuggeranong': 'Sur',
 'MountGinini': 'Sur',
 'Ballarat': 'Sur',
 'Bendigo': 'Sur',
 'Sale': 'Sur',
 'MelbourneAirport': 'Sur',
 'Melbourne': 'Sur',
 'Mildura': 'Sur',
 'Nhil': 'Sur',
 'Portland': 'Sur',
 'Watsonia': 'Sur',
 'Dartmoor': 'Sur',
 'Brisbane': 'Este',
 'Cairns': 'Norte',
 'GoldCoast': 'Este',
 'Townsville': 'Norte',
 'Adelaide': 'Sur',
 'MountGambier': 'Sur',
 'Nuriootpa': 'Sur',
 'Woomera': 'Sur',
 'Albany': 'Oeste',
 'Witchcliffe': 'Oeste',
 'PearceRAAF': 'Norte',
 'PerthAirport': 'Oeste',
 'Perth': 'Oeste',
 'SalmonGums': 'Oeste',
 'Walpole': 'Oeste',
 'Hobart': 'Sur',
 'Launceston': 'Sur',
 'AliceSprings': 'Norte',
 'Darwin': 'Norte',
 'Katherine': 'Norte',
 'Uluru': 'Norte'}

def map_location_to_region(X, mapping):
    """Mapea la columna Location a Regiones usando un diccionario."""
    
    # X llega como DataFrame (n_samples, 1), lo convertimos a pd.Series
    X_col = X.iloc[:, 0]
    
    # .map() aplica el diccionario.
    # Si una 'Location' no está en el 'mapping', pondrá NaN.
    regions = X_col.map(mapping)
    
    # Devolver como (n_samples, 1) para sklearn
    return regions.values.reshape(-1, 1)

def get_region_feature_names(transformer, input_features):
    return ['Regiones']


pipeline = joblib.load('pipeline.pkl')

logger.info('loaded pipeline')

df_input = pd.read_csv('/files/input.csv')

logger.info('loaded input')

print("shape:", df_input.shape)
print(df_input.head())

output = pipeline.predict(df_input)

logger.info('made predictions')

pd.DataFrame(output, columns=['Rain_predicted']).to_csv('/files/output.csv', index=False)

logger.info('saved output')