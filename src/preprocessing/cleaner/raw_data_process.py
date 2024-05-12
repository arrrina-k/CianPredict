"""
Makes raw_cian_data clean without NaN and "-1" values;
Adds coordinates_of_underground for each example
"""

import ast

import pandas as pd
import requests


class DataProcess:
    def __init__(self, raw_path, interim_path, processed_path, yandex_apikey=None):
        """
        Initialization for DataProcess
        :param raw_path: - path to the raw file from scrapper script
        :param interim_path: - path to the file, where you want to save interim file
        :param processed_path: - path to the file, where you want to store final data
        :param yandex_apikey: - user need to get it from https://developer.tech.yandex.ru/services/
        """
        self.data_raw_path = raw_path
        self.data_interim_path = interim_path
        self.data_processed_path = processed_path
        self.apikey = yandex_apikey

    @staticmethod
    def __fetch_coordinates(apikey, address) -> tuple or None:
        """
        provides coordinates for each example
        :param apikey:
        :param address:
        :return:
        """
        base_url = "https://geocode-maps.yandex.ru/1.x"
        response = requests.get(
            base_url,
            params={
                "geocode": address,
                "apikey": apikey,
                "format": "json",
            },
        )
        response.raise_for_status()
        found_places = response.json()["response"]["GeoObjectCollection"]["featureMember"]

        if not found_places:
            return None

        most_relevant = found_places[0]
        lon, lat = most_relevant["GeoObject"]["Point"]["pos"].split(" ")
        return lon, lat

    def run_interim(self) -> None:
        """
        process of cleaning raw_cian_data and getting coordinates of undergrounds
        :return None
        """
        df = pd.read_csv(self.data_path)
        df = df.drop_duplicates().drop(
            columns=[
                "location",
                "deal_type",
                "accommodation_type",
                "url",
                "price_per_month",
                "commissions",
                "author",
                "house_number",
                "residential_complex",
            ]
        )
        indexes = df.loc[df["floors_count"] == -1].index
        df = df.drop(indexes)
        indexes = df.loc[df["total_meters"] <= 9.00].index
        df = df.drop(indexes)
        df = df.dropna(subset="price")
        indexes = df.loc[df["rooms_count"] == -1].index
        df = df.drop(indexes)
        indexes = df.loc[df["floor"] == -1].index
        df = df.drop(indexes)
        indexes = df[(df["underground"].isna()) & (df["district"].isna())].index
        df = df.drop(indexes)
        df = df[df.notna().sum(axis=1) == 9]
        coords = list()
        for underground in df["underground"]:
            coords.append(self.__fetch_coordinates(self.apikey, underground))
        df["coordinates_of_underground"] = coords
        df.to_csv(self.data_interim_path, index=False)

    def run_processed(self) -> None:

        def rename(x):
            if x == "real_estate_agent":
                value = "Агент по недвижимости"
            elif x == "developer":
                value = "Застройщик"
            elif x == "homeowner":
                value = "Владелец"
            elif x == "realtor":
                value = "Риэлтор"
            elif x == "official_representative":
                value = "Официальный представитель"
            elif x == "representative_developer":
                value = "Титулованный застройщик"
            else:
                value = "Неизвестно"

            return value

        data = pd.read_csv(self.data_interim_path)

        lons = []
        lats = []
        for row in data.iterrows():
            lon, lat = ast.literal_eval(row[1].iloc[-1])
            lons.append(float(lat))
            lats.append(float(lon))

        data["lons"] = pd.Series(lons)
        data["lats"] = pd.Series(lats)

        data = data.drop(columns=["coordinates_of_underground"])
        data["price"] = (data["price"] / 1000000).round(2)
        data["author_type"] = data["author_type"].apply(rename)

        data.to_csv(self.data_processed_path, index=False)
