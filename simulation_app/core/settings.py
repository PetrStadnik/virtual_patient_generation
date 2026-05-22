import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import certifi
os.environ.setdefault('SSL_CERT_FILE', certifi.where())
os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())

BASE_DIR = Path(__file__).resolve().parent.parent
# Make the project root (parent of simulation_app/) importable so that
# "from simulation_app.sim.runner import ..." works inside Django too.
PROJECT_ROOT = str(BASE_DIR.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(BASE_DIR.parent / '.env')

# Optional: path to a FHIR bundle JSON file used as the default scenario.
# Set via environment variable SIMULATION_BUNDLE or directly here.
SIMULATION_BUNDLE = BASE_DIR.parent / "scenarios/multiagent/anthropic/A09.0_9_ŽofieNěmcová.json"
#SIMULATION_BUNDLE: str = os.getenv("SIMULATION_BUNDLE", "")

SECRET_KEY = 'dev-only-secret-key'
DEBUG = True
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.staticfiles',
    'sim',
]

TEMPLATES = [{
    'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'DIRS': [],
    'APP_DIRS': True,
    'OPTIONS': {'context_processors': ['django.template.context_processors.request']},
}]

ROOT_URLCONF = 'core.urls'
STATIC_URL = '/static/'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
