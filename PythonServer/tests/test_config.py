from app.core.config import settings


def test_settings_has_expected_directories():
    assert settings.AUDIO_OUTPUT_DIR == "generated_audio"
    assert settings.DATASETS_DIR == "datasets"
    assert settings.PACKAGE_DIR == "packages"


def test_settings_default_sample_rate_is_supported():
    assert settings.DEFAULT_SAMPLE_RATE in settings.SAMPLE_RATES
