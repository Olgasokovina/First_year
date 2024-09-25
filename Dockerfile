# Соберём образ контейнера и назовём его server_image_sl:
    #      docker build -t server_image_sl .

# Затем запустим контейнер:
    # docker run -it --rm --name=server_container -p=5000:5000 server_image_sl


# Задаём базовый образ
FROM python:3.8-slim

# Копируем  вспомогательные файлы в рабочую директорию контейнера

COPY ./project/data/crime.pkl ./
COPY ./project/data/life_level.pkl ./
COPY ./project/data/public_school_rankings.pkl ./
COPY ./project/data/capital_state.pkl ./
COPY ./project/data/bg_city.pkl ./
COPY ./project/data/lg_city.pkl ./
COPY ./project/data/mln_city.pkl ./
COPY ./project/data/city_center.pkl ./

COPY ./app/models/best_model.pkl ./
COPY ./app/models/automl.pkl ./
COPY ./app/models/selector_RFECV.pkl ./
COPY ./app/models/model_GaussianMixture.pkl ./
COPY ./app/server.py ./

COPY ./requirements.txt ./
COPY ./func_for_server.py ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "./server.py" ]