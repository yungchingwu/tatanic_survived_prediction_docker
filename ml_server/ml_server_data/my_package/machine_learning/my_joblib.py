import joblib


def save_job(job, file_name: str, path: str):
    full_path = f"{path}{file_name}.joblib"
    joblib.dump(job, full_path)

def load_job(file_name: str, path: str):
    full_path = f"{path}{file_name}.joblib"
    return joblib.load(full_path)
