"""
collects data from https://www.cian.ru
"""

import csv

import cianparser


class Scraper:
    def __init__(self, additional_settings, city, file_path):
        """
        :param additional_settings(
        object_type - тип жилья ("new" - вторичка, "secondary" - новостройка)
        start_page - страница, с которого начинается сбор данных
        end_page - страница, с которого заканчивается сбор данных
        is_by_homeowner - объявления, созданных только собственниками
        min_price - цена от (в рублях)
        max_price - цена до (в рублях)
        min_balconies - минимальное количество балконов
        have_loggia - наличие лоджи
        min_house_year - год постройки дома от
        max_house_year - год постройки дома до
        min_floor - этаж от
        max_floor - этаж до
        min_total_floor - этажей в доме от
        max_total_floor - этажей в доме до
        house_material_type - тип дома
        metro - название метрополитена
        metro_station - станция метро (доступно при заданом metro)
        metro_foot_minute - сколько минут до метро пешком
        flat_share - с долями или без (1 - только доли, 2 - без долей)
        only_flat - без апартаментов
        only_apartment - только апартаменты
        sort_by - сортировка объявлений
        ):
        :param city: - like "Москва"
        :param file_path: - path to an existing csv to write in
        """
        self.__execute_file(additional_settings, city, file_path)

    @staticmethod
    def __execute_file(additional_settings, city, file_path):
        """
        executes the file filling in
        :param additional_settings:
        :param city:
        :param file_path:
        :return:
        """
        moscow_parser = cianparser.CianParser(location=city)
        for num in range(5):
            data = moscow_parser.get_flats(
                deal_type="sale",
                rooms=num,
                with_saving_csv=False,
                additional_settings=additional_settings,
                with_extra_data=False,
            )
            fieldnames = data[0].keys()

            with open(file_path, "a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)

                if file.tell() == 0:
                    writer.writeheader()

                for row in data:
                    writer.writerow(row)
