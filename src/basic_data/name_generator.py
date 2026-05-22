from typing import Tuple, Literal
from babel import Locale
from babel.languages import get_official_languages
from faker import Faker

def _country_to_locale(country_code: str) -> str:
    """CZ -> cs_CZ, DE -> de_DE, CH -> de_CH, ..."""
    cc = country_code.upper()
    langs = get_official_languages(cc, regional=False, de_facto=True)
    lang = langs[0]
    locale = f"{lang}_{cc}"
    return locale

def get_name_and_surname(gender: Literal["F", "M"], country_code: str) -> Tuple[str, str]:
    fake = Faker(_country_to_locale(country_code))
    if gender == "F":
        return fake.first_name_female(), fake.last_name_female()
    else:
        return fake.first_name_male(), fake.last_name_male()


