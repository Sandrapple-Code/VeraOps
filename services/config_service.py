import os
import json
from typing import Any, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_PATH = os.path.join(PROJECT_ROOT, "settings.json")

DEFAULT_SETTINGS: Dict[str, Any] = {
    "groq_api_key": "",
    "model_selection": "llama-3.3-70b-versatile",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k_retrieval": 3,
    "temperature": 0.1
}

def load_env() -> None:
    """
    Loads environment variables from a .env file in the project root if it exists.
    """
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        val = val.strip()
                        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                            val = val[1:-1]
                        os.environ[key.strip()] = val
        except Exception:
            pass

def load_settings() -> Dict[str, Any]:
    """
    Loads configuration settings from settings.json.
    Falls back to environment variables for API Key if not set.
    """
    load_env()
    settings = DEFAULT_SETTINGS.copy()
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
                settings.update(saved)
        except Exception:
            pass
            
    # Check environment variable for Groq API Key if settings file is empty
    if not settings.get("groq_api_key"):
        env_key = os.environ.get("GROQ_API_KEY", "")
        if env_key:
            settings["groq_api_key"] = env_key
            
    return settings

def save_settings(settings: Dict[str, Any]) -> None:
    """
    Saves settings dict to settings.json.
    """
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        pass

def get_setting(key: str, default: Any = None) -> Any:
    """
    Retrieves a single setting value.
    """
    settings = load_settings()
    # Ensure types are correct
    val = settings.get(key, default)
    if key in DEFAULT_SETTINGS:
        expected_type = type(DEFAULT_SETTINGS[key])
        try:
            val = expected_type(val)
        except (ValueError, TypeError):
            val = DEFAULT_SETTINGS[key]
    return val

def set_setting(key: str, value: Any) -> None:
    """
    Sets and saves a single setting value.
    """
    settings = load_settings()
    settings[key] = value
    save_settings(settings)
