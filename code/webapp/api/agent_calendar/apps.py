from django.apps import AppConfig


class AgentCalendarConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "api.agent_calendar"

    def ready(self) -> None:
        from api.general.utils.running import is_main_running

        if not is_main_running():
            return

        # Load JSON files into memory
        from .startup import load_data

        # Load JSON files into memory
        load_data()
