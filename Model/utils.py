import os
import atexit
from apscheduler.schedulers.background import BackgroundScheduler

def delete_user_images():
    folder_path = 'user_images'
    file_list = os.listdir(folder_path)
    if file_list:
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            try:
                os.remove(file_path)
                print('File deleted:', file_path)
            except OSError as e:
                print('Error deleting file:', file_path, e)

def load_delete_user_images_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=delete_user_images, trigger="interval", seconds= 1 * 60 * 60)
    scheduler.start()

    atexit.register(lambda: scheduler.shutdown())
