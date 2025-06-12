from django.apps import AppConfig


class GeneralConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "api.general"

    def ready(self) -> None:
        from api.general.utils.running import is_main_running

        if not is_main_running():
            return

        # Load JSON files into memory
        from .startup import load_data

        load_data()
