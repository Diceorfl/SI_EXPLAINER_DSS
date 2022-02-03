from reportlab.pdfgen import canvas
from reportlab.platypus import Table
from reportlab.platypus import TableStyle
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from io import BytesIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


def pca_plot(df: pd.DataFrame) -> bytes:
    clusters = "Clusters"
    x = df.loc[:, df.columns != clusters].values
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    components = pca.fit_transform(x)
    total_var = pca.explained_variance_ratio_.sum() * 100

    fig = px.scatter(components, x=0, y=1, color=df[clusters],
                     title=f'Total Explained Variance: {total_var:.2f}%', labels={'0': 'PC 1', '1': 'PC 2'})

    return fig.to_image(format="jpg")


def create_table(table_lines: list) -> Table:
    for i, line in enumerate(table_lines):
        table_lines[i][1] = '\n'.join(table_lines[i][1][j:j + 100] for j in range(0, len(table_lines[i][1]), 100))

    table = Table(table_lines)
    style = TableStyle([
        ('BACKGROUND', (0, 0), (2, 0), colors.blue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Roman'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BOX', (0, 0), (-1, -1), 2, colors.black),
        ('GRID', (0, 0), (-1, -1), 2, colors.black)
    ])
    table.setStyle(style)

    return table


def write_text(pdf: canvas.Canvas, x: int, y: int, text_lines: list):
    text = pdf.beginText(x, y)
    text.setFont("Times-Roman", 14)
    for line in text_lines:
        for ll in line.split("\n"):
            ll = '\n'.join(ll[j:j + 95] for j in range(0, len(ll), 95))
            for l in ll.split("\n"):
                text.textLine(l.strip())
                x, y = text.getCursor()
                if y <= 5:
                    pdf.drawText(text)
                    pdf.showPage()
                    text = pdf.beginText(50, 800)
                    text.setFont("Times-Roman", 14)
    pdf.drawText(text)


def general_info(df: pd.DataFrame, continuous: list, categorical: list) -> list:
    text = []
    for feature in continuous:
        cluster_avg = df.groupby('Clusters').agg({feature: ['mean']})
        text.append("Minimum mean value of %s: %s in cluster %s\n" %
                    (feature, cluster_avg[feature].min()[0], cluster_avg[feature].idxmin()[0]))
        text.append("Maximum mean value of %s: %s in cluster %s\n" %
                    (feature, cluster_avg[feature].max()[0], cluster_avg[feature].idxmax()[0]))

    for feature in categorical:
        cluster_mode = df.groupby('Clusters').agg(lambda x: x.value_counts().index[0])[feature]
        for i in cluster_mode.index:
            text.append("More often value of %s: %s in cluster %s\n" % (feature, cluster_mode.loc[i], i))

    return text


def show_general_info(pdf: canvas.Canvas, df: pd.DataFrame, continuous: list, categorical: list):
    y = 800
    for feature in continuous:
        y -= 260
        if y < 0:
            y = 800 - 260
            pdf.showPage()
        fig = px.histogram(df, x='Clusters', y=feature, histfunc="avg", nbins=len(df["Clusters"].unique()) * 2, text_auto=True)
        hist_img = ImageReader(BytesIO(fig.to_image(format="jpg")))
        pdf.drawImage(hist_img, 100, y, 400, 250)
        plt.clf()

    for feature in categorical:
        y -= 260
        if y < 0:
            y = 800 - 260
            pdf.showPage()
        fig = px.histogram(df, x='Clusters', category_orders=dict(clusters=df['Clusters'].unique()),
                           color='Clusters', pattern_shape=feature, nbins=len(df["Clusters"].unique()) * 2)
        hist_img = ImageReader(BytesIO(fig.to_image(format="jpg")))
        pdf.drawImage(hist_img, 100, y, 400, 250)
        plt.clf()

    text_lines = general_info(df, continuous, categorical)
    write_text(pdf, 40, y-20, text_lines)


def find_differences(df: pd.DataFrame, differences: dict, continuous: list, categorical: list) -> dict:
    clusters = "Clusters"
    result = {str(el): [] for el in set(df[clusters])}

    for key in differences.keys():
        cluster1, cluster2 = str(key[0]), str(key[1])
        msg = f'Patients in cluster {cluster1} differ from patients in cluster {cluster2} in the following features:\n'
        for feature in differences[key]:
            feature = str(feature)
            if feature in continuous:
                average_value0 = np.mean(df[df[clusters] == cluster1][feature])
                average_value1 = np.mean(df[df[clusters] == cluster2][feature])
                msg += f'{feature}: where the mean value for patients in cluster {cluster1} is {average_value0:.3f} and ' \
                       f'for patients in cluster {cluster2} is {average_value1:.3f}.\n'
            if feature in categorical:
                mode0 = df[df[clusters] == cluster1][feature].mode()[0]
                mode1 = df[df[clusters] == cluster2][feature].mode()[0]
                msg += f'{feature}: where is the most characteristic value for patients in the cluster {cluster1} is {mode0} ' \
                       f'and for patients in cluster {cluster1} is {mode1}.\n'
        result[cluster1].append(msg)
        result[cluster2].append(msg)

    return result


def create_clusters_info(pdf: canvas.Canvas, df: pd.DataFrame, significant_features: list, differences: dict,
                         continuous: list, categorical: list):
    diff = find_differences(df.copy(), differences, continuous, categorical)
    for feature in significant_features[1:]:
        pdf.showPage()
        pdf.setFont("Times-Bold", 14)
        pdf.drawCentredString(100, 800, "Cluster " + feature[0] + ":")
        y = 800
        for f in feature[1].split(", "):
            y -= 260
            if y < 0:
                y = 800 - 260
                pdf.showPage()
            f = f.replace("\n", "")
            if f in continuous:
                fig = px.histogram(df[df["Clusters"] == feature[0]], x=f, marginal='box')
                hist_plot_img = ImageReader(BytesIO(fig.to_image(format="jpg")))
                pdf.drawImage(hist_plot_img, 100, y, 400, 250)
            if f in categorical:
                fig = px.histogram(df[df["Clusters"] == feature[0]], x=f, category_orders=dict(clusters=df[f].unique()),
                                   nbins=len(df[f].unique())*3, text_auto=True)
                hist_plot_img = ImageReader(BytesIO(fig.to_image(format="jpg")))
                pdf.drawImage(hist_plot_img, 100, y, 400, 250)
            plt.clf()
        write_text(pdf, 40, y - 20, diff[feature[0]])


class InterpretationReport(object):

    def __init__(self, buffer: BytesIO):
        self._pdf = canvas.Canvas(buffer)
        self._pdf.setTitle("Interpretation results")
        self._file_name, self._table_lines = None, None
        self._continuous, self._categorical = [], []
        self._df = pd.DataFrame()
        self._differences = {}

    def set_file_name(self, file_name: str):
        self._file_name = file_name

    def set_table_lines(self, table_lines: list):
        self._table_lines = table_lines

    def set_differences(self, differences: list):
        self._differences = differences

    def set_data(self, data: pd.DataFrame):
        self._df = data

    def set_features(self, continuous: list, categorical: list):
        self._continuous, self._categorical = continuous, categorical

    def create_report(self):
        self._pdf.setFont("Times-Bold", 16)
        self._pdf.drawCentredString(300, 780, "Interpretation results of " + self._file_name)
        pca_img = ImageReader(BytesIO(pca_plot(self._df)))
        self._pdf.drawImage(pca_img, 100, 375, 400, 400)
        plt.clf()
        text_lines = ["The results of the analysis showed that %s patients were divided into "
                      "%s groups as follows:\n" % (self._df["Clusters"].count(), self._df["Clusters"].nunique())]
        write_text(self._pdf, 40, 365, text_lines)
        fig = px.histogram(self._df, x="Clusters", category_orders=dict(clustert=self._df["Clusters"].unique()),
                           nbins=len(self._df["Clusters"].unique()), text_auto=True)
        hist_img = ImageReader(BytesIO(fig.to_image(format="jpg")))
        self._pdf.drawImage(hist_img, 100, 50, 400, 300)
        plt.clf()
        self._pdf.showPage()
        show_general_info(self._pdf, self._df, self._continuous, self._categorical)
        self._pdf.showPage()
        table = create_table(self._table_lines[:])
        table.wrapOn(self._pdf, 400, 100)
        table.drawOn(self._pdf, 300 - table._width / 2, 800 - table._height)
        create_clusters_info(self._pdf, self._df, self._table_lines, self._differences, self._continuous, self._categorical)
        self._pdf.save()
