import requests
import os
from datetime import datetime, timedelta

# Ustawienia linku i katalogu docelowego
base_url = "https://retsuz.pl/tracks/tracking/"
output_directory = "pictures"

# Utwórz katalog, jeśli nie istnieje
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Ustaw daty początkową i końcową
start_date = datetime.strptime('202401071200', '%Y%m%d%H%M')
end_date = datetime.strptime('202401080415', '%Y%m%d%H%M')

# Pobieranie plików dla każdej daty z zakresu
current_date = start_date
while current_date <= end_date:
    filename = f"{current_date.strftime('%Y%m%d%H%M')}.png"
    url = base_url + filename

    # Pobieranie pliku
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(output_directory, filename), 'wb') as f:
            f.write(response.content)
        print(f"Pobrano: {filename}")
    else:
        print(f"Błąd pobierania: {filename}")

    # Zwiększ datę o 5 minut
    current_date += timedelta(minutes=5)