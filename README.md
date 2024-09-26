# Final_1_year  

## Оглавление  
[1. Описание проекта](#описание-проекта)  
[2. Какой кейс решаем](#какаой-кейс-решаем)  
[3. Краткая информация о данных](#краткая-информация-о-данных)  
[4. Этапы работы над проектом](#этапы-работы-над-проектом)  
[5. Метод решения задачи](#метод-решения-задачи)  
[6. Результат](#резульат)  
[7. Как можно улучшить?](#Как-можно-улучшить-проект?)  

### Описание проекта  
Агентство недвижимости. Задача — разработать сервис для предсказания стоимости домов на основе истории предложений.

### Какаой кейс решаем  
Что практикуем:  
* Преобразование данных  
* Разведочный анализ  
* Очистка данных  
* Подбор модели машинного обучения, подбор гиперпараметров выбранной модели.  
* Реализации предложенного решения в продакшене.  
* Учимся писать хороший (красивый) код на Python, в соответствии с PEP8.  

### Краткая информация о данных  
В распоряжении будет база "сырых" данных.  
Проблематика: В представленных данных нет ни одного числового столбца, все данные перепутаны по столбцам, много опечаток и недостоверных данных, нет единой системы единиц при измерении площади, некоторые данные противоречивы. Крайне скудное описание данных.  

### Этапы работы над проектом   
* Внимательно изучить данные. Привести (перекодировать) данные в числовой вид.  
* Провести очистку данных.  
* Добавить новые данные из открытых источников.  
* Провести разведочный анализ данных.  
* Выбрать наиболее подходящую модель машинного обучения. Провести подбор гиперпараметров. Поверить воспроизводимость модели - как модель работает на новых данных.  
* Реализовать решение в продакшене. Создать докер образ.  
* Загрузить ноутбук со своим решением на GitHub, оформив его в соответствии с требованиями.  

### Метод решения задачи  
Скачать ноутбук с данными, оценить какие данные в нем находятся и в каком виде. Написать вспомогательные функции для вычленения необходимых данных На основании этих данных создать новые признаки. Нормализовать полученные и имеющиеся данные. Удалить ненужные, избыточные, мусорные данные. Попытаться с помощью мамематических методов и здравого смысла найти и удалить "выбросы". Заполнить, по возможности, (или подходящей мерой центральной тенденции) недостающие данные. Визуализировать полученный результат. При этом код должен быть читаемым и понятным: имена переменных и функций отражать их сущность, избегать многострочных конструкций и условий.  
Создать новые признаки, улучшающие качество модели.  

### Резульат  
Ссылка на проект в GitHub с выполненными заданиями, выводами и графиками. Также прописан докерфайл и можно запускать предсказания из контейнера.  
docker build -t server_image_sl . # создать контейнер  
docker run -it --rm --name=server_container -p=5000:5000 server_image_sl # Запустить контейнер  

test\test_.ipynb  # ноутбук для предсказания стоимости. Необходимозапустить контейнер и ввести необходимые данные и далее отправить пост запрос.  


### Как можно улучшить проект?  
В своем проекте я не использовала 'street' - можно поробовать поработать с этими данными, например создать признак - удаленность от центра города (downtown). Можно добавить координаты интересных мест притяжения - памятники, музеи, концертные залы и т.д. (попробовать найти их координаты)  и проверить как влияет удаленность на цену недвижимости.  
Можно более тщательно поработать с данными по школам, тк на мой взгляд эти данные содержат много недостоверной информации. Можно создать признаки по level school: elementary, middle, high, preschool, kindergarden, наличию/отсутствию частных школ. Возможно это улучшит качество моделей.  
Можно попробовать изменить признаки "sqft" и "lotsize". Тк эти признаки имеют большой вес в модели, то возможно улучшится качество модели.  
Безусловно можно сделать более "жесткую" очистку данных, но тогда может оказаться крайне мало данных для обучения. И необходимо вводить жесткие ограничения для параметров недвижимости, поступающих в модель.  
