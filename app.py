import base64
import io
from io import BytesIO
import plotly.express as px

import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from dash import dash_table
import dash_bootstrap_components as dbc

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from clustering_interpretation import ClusteringInterpretation
from report import InterpretationReport

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

colors = {"graphBackground": "#F5F5F5", "background": "#ffffff", "text": "#000000"}

data = pd.DataFrame()
interpretation_df = None
buffer = BytesIO()
interpretation_report = InterpretationReport(buffer)

app.layout = html.Div(
    [
        html.H1("РЕЗУЛЬТАТЫ ИНТЕРПРЕТАЦИИ", style={'text-align': 'center'}),
        html.Div(id="load", children=[
            html.Div(id="upload", children=[
                dcc.Upload(
                    id="upload_data",
                    children=html.Div(["Перетащите или ", html.A("Выберите файл")]),
                    style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "textAlign": "center",
                        "margin-bottom": "10px",
                    },
                    # Allow multiple files to be uploaded
                    multiple=True,
                )], style=dict(width='90%', display='inline-block')),
            html.Div(id="download", children=[
                html.Button("Скачать PDF", id="btn_pdf",
                            disabled=True,
                            n_clicks=0,
                            style={
                                "width": "200px",
                                "height": "60px",
                                "margin-left": "10px"
                            }),
                dcc.Download(id="download_report_pdf"),
            ], style=dict(width='14%', display='inline-block')),
        ], style=dict(display='flex')),

        html.Div(id='dataset_info', children=[
            html.Div(id='continuous_info', children=[
                html.Label('Непрерывные'),
                dcc.Dropdown(
                    id='continuous',
                    clearable=False,
                    searchable=False,
                    multi=True,
                    style={
                        "width": "100%"
                    }
                )], style=dict(width='35%')),
            html.Div(id='categorical_info', children=[
                html.Label('Категориальные'),
                dcc.Dropdown(
                    id='categorical',
                    clearable=False,
                    searchable=False,
                    multi=True,
                    style={
                        "width": "100%"
                    }
                ),
                html.Button(
                    'ПОДТВЕРДИТЬ',
                    id='submit_button',
                    n_clicks=0,
                    disabled=True,
                    style={
                        "margin-top": "20px",
                        "width": "100%"
                    }
                ),
                dbc.Tooltip(
                    f"Если вся информация указана верно, то нажмите подтвердить, "
                    f"иначе внесите изменения и подтвердите их.",
                    target=f"submit_button",
                    placement="top",
                ),
            ], style={"width": "35%", "margin-left": "30px"}),
            html.Div(id='clusters_info', children=[
                html.Label('Кластеры'),
                dcc.Dropdown(
                    id='clusters',
                    clearable=False,
                    searchable=False,
                    multi=True,
                    style={
                        "width": "100%"
                    }
                )], style={"width": "10%", "margin-left": "30px"}),
            html.Div(id='dependent_info', children=[
                html.Label('Являются ли сравниваемые группы зависимыми?'),
                dcc.RadioItems(
                    id="dependent",
                    options=[
                        {'label': 'Да', 'value': 'True'},
                        {'label': 'Нет', 'value': 'False'},
                    ],
                    value='False',
                    style={
                        "width": "100%"
                    },
                    labelStyle={'display': 'inline-block'}
                ),
                dbc.Tooltip(
                    f"Например, если измерения признаков производились несколько раз у пациентов. "
                    f"(Уровень сахара в крови/вес/давление)",
                    target=f"dependent",
                    placement="auto-end",
                )], style={"width": "20%", "margin-left": "40px"}),
        ], style={"display": 'flex'}),

        html.Div(id='mode', children=[
            html.Label('Режим аналитики:'),
            dcc.Dropdown(
                id='set_mode',
                value='None',
                clearable=False,
                searchable=False,
                multi=False,
                style={"textAlign": "center"}
            ),
        ]),

        html.Div(id="main_table"),

        html.Div(id="why", children=[
            dcc.Textarea(
                id='ask_textarea',
                value='Здесь вы можете спросить: почему пациент с ID: _  в кластере: _ или '
                      'почему пациент с ID: _  не в кластере: _',
                style={'width': '90%', 'height': "60px", "margin-top": "10px"},
            ),
            html.Button('Спросить', id='ask_button',
                        n_clicks=0,
                        disabled=True,
                        style={
                            'width': '200px',
                            'height': "65px",
                            "margin-left": "10px",
                            "margin-top": "10px",
                            "textAlign": "center",
                        }
                        ),
        ], style={"display": "flex"}),

        html.Div(id="plots"),
        html.Div(id="significant_features_table")
    ]
)


def clear():
    global data, interpretation_df, buffer, interpretation_report
    data = pd.DataFrame()
    interpretation_df = None
    buffer = BytesIO()
    interpretation_report = InterpretationReport(buffer)


def parse_data(contents, filename):
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif "xls" in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif "txt" or "tsv" in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), delimiter=r"\s+")
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])

    return df


def pca_plot(df: pd.DataFrame, clusters: str) -> px.scatter():
    x = df.loc[:, df.columns != clusters].values
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    components = pca.fit_transform(x)
    total_var = pca.explained_variance_ratio_.sum() * 100

    fig = px.scatter(components, x=0, y=1, color=df[clusters],
                     title=f'Общая "объяснительная" дисперсия: {total_var:.2f}%', labels={'0': 'PC 1', '1': 'PC 2'})

    return fig


def create_avg_plots(df: pd.DataFrame, continuous: list, categorical: list, clusters: str) -> list:
    plots = []
    for feature in continuous:
        fig = px.histogram(df, x=clusters, y=feature, color=clusters, histfunc="avg",
                           nbins=len(df[clusters].unique()) * 2, text_auto=True)
        plots.append(fig)

    for feature in categorical:
        fig = px.histogram(df, x=clusters, category_orders=dict(cluster=df[clusters].unique()),
                           color=clusters, pattern_shape=feature, nbins=len(df[clusters].unique()) * 2)
        plots.append(fig)

    return plots


def find_differences(cluster: str, df: pd.DataFrame, continuous: list, categorical: list, clusters: str) -> list:
    differences = interpretation_df.get_differences()
    result = []
    for key in differences.keys():
        if str(key[0]) != str(cluster) and str(key[1]) != str(cluster):
            continue
        msg = f'''#### **Пациенты в кластере {key[0]} отличаются от пациентов в кластере {key[1]} по следующим признакам:**\n'''
        for feature in differences[key]:
            if feature in continuous:
                average_value0 = np.mean(df[df[clusters] == key[0]][feature])
                average_value1 = np.mean(df[df[clusters] == key[1]][feature])
                msg += f'''#### *{feature}*: где среднее значение для пациентов в кластере *{key[0]}* равно {average_value0:.3f}, а ''' \
                       f'''для пациентов в кластере *{key[1]}* равно {average_value1:.3f}.\n'''
            if feature in categorical:
                mode0 = df[df[clusters] == key[0]][feature].mode()[0]
                mode1 = df[df[clusters] == key[1]][feature].mode()[0]
                msg += f'''#### *{feature}*: где наиболее характерное значение для пациентов в кластере *{key[0]}* равно {mode0}, а ''' \
                       f'''для пациентов в кластере *{key[1]}* равно {mode1}.\n'''
        msg += '''\n'''
        result.append(msg)

    return [dcc.Markdown("".join(result))]


def find_significant_features(cluster: str, df: pd.DataFrame, continuous: list, categorical: list,
                              clusters: str) -> list:
    significant_features = interpretation_df.get_significant_features()
    figures = []
    for feature in significant_features[cluster]:
        fig = None
        if feature in continuous:
            fig = px.histogram(df[df[clusters] == cluster], x=feature, marginal='box')
        if feature in categorical:
            fig = px.histogram(df[df[clusters] == cluster], x=feature,
                               category_orders=dict(feature=df[feature].unique()),
                               nbins=len(df[feature].unique()) * 2, color=feature, text_auto=True)
        figures.append(fig)

    return [dcc.Graph(id="fig" + str(i), figure=fig) for i, fig in enumerate(figures)]


def cluster_info(cluster: str, df: pd.DataFrame, continuous: list, categorical: list, clusters: str) -> list:
    return find_significant_features(cluster, df, continuous, categorical, clusters) + \
           find_differences(cluster, df, continuous, categorical, clusters)


def find_categorical(df: pd.DataFrame) -> dict:
    categorical_features = {feature: False for feature in df.columns}
    for feature in df.columns:
        unique_percent = df[feature].nunique() / len(df[feature])
        if unique_percent <= 0.1:
            categorical_features[feature] = True

    return categorical_features


def distribute_features_automatically(df: pd.DataFrame) -> dict:
    categorical_features = find_categorical(df)
    cat_features = []
    noncat_features = []
    for feature in categorical_features.keys():
        if categorical_features[feature]:
            cat_features.append(feature)
        else:
            noncat_features.append(feature)

    output = {"Continuous": noncat_features,
              "Categorical": cat_features[:-1],
              "Clusters": [cat_features[-1]]}

    return output


def create_interpretation_df(continuous: list, categorical: list, clusters: str, dependent: bool):
    global interpretation_df
    interpretation_df = ClusteringInterpretation(data.copy(), clusters)
    interpretation_df.set_continuous_and_categorical(continuous=continuous, categorical=categorical)
    interpretation_df.set_dependent_clusters(True if dependent == "True" else False)


@app.callback(
    [
        Output("continuous", "options"),
        Output("continuous", "value"),
        Output("categorical", "options"),
        Output("categorical", "value"),
        Output("clusters", "options"),
        Output("clusters", "value"),
        Output("submit_button", "disabled")
    ],
    [
        Input("upload_data", "contents"), Input("upload_data", "filename")
    ],
)
def update_dropdowns_automatically(contents, filename):
    clear()

    global data

    options = [{'label': column, 'value': column} for column in [""]]
    values = {"Continuous": "", "Categorical": "", "Clusters": ""}
    disabled = True

    if contents:
        contents = contents[0]
        filename = filename[0]
        interpretation_report.set_file_name(filename)
        df = parse_data(contents, filename)
        data = df.copy()
        options = [{'label': column, 'value': column} for column in df.columns]
        values = distribute_features_automatically(df)
        disabled = False

    return [
            options, values["Continuous"],
            options, values["Categorical"],
            options, values["Clusters"],
            disabled
    ]


@app.callback(
    [
        Output("set_mode", "options"),
        Output("set_mode", "value")
    ],
    Input('submit_button', 'n_clicks'),
    State('clusters', 'value')
)
def update_mode(n_clicks, clusters):
    options = [{"label": "Все кластеры", "value": "None"}]
    if n_clicks > 0:
        clusters = clusters[-1]
        options = [{"label": "Кластер %s" % clstr, "value": clstr} for clstr in data[clusters].unique()]
        options.append({"label": "Все кластеры", "value": "all"})
        return options, "all"

    return options, "None"


@app.callback(
    Output("main_table", "children"),
    Input('submit_button', 'n_clicks')
)
def update_main_table(n_clicks):
    if n_clicks > 0:
        df = data.copy()
        df.insert(loc=0, column="ID", value=[x + 1 for x in range(len(df.index))])

        children = [
            dash_table.DataTable(
                data=df.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df.columns],
                page_size=6,  # number of rows visible per page
                style_table={"margin-top": "10px"},
                style_cell={
                    'minWidth': 95,
                    'maxWidth': 95,
                    'width': 95,
                    'textAlign': 'center'
                },
            ),
        ]

        return children


@app.callback(
    Output('ask_button', 'disabled'),
    Input('submit_button', 'n_clicks'),
)
def update_why(n_clicks):
    if n_clicks > 0:
        return False
    return True


@app.callback(
    Output('ask_textarea', 'value'),
    Input('ask_button', 'n_clicks'),
    [
        State('continuous', 'value'),
        State('categorical', 'value'),
        State('clusters', 'value'),
        State('ask_textarea', 'value')
    ]
)
def update_answer(n_clicks, continuous, categorical, clusters, text):
    answer = 'Здесь вы можете спросить: почему пациент с ID: _  в кластере: _ или ' \
             'почему пациент с ID: _  не в кластере: _'
    if n_clicks > 0:
        try:
            text = text.split()
            not_flag = ""
            id_patient, cluster = [int(s) for s in text if s.isdigit()][0] - 1, text[-1]
            if cluster.isdigit():
                cluster = int(cluster)
            if "не" in text:
                not_flag = "не"
        except Exception as e:
            return answer

        if id_patient not in data.index or cluster not in list(data[clusters[0]]) or (
                not_flag == "" and cluster not in list(data.iloc[[id_patient]][clusters[0]])) \
                or (not_flag == "не" and cluster in list(data.iloc[[id_patient]][clusters[0]])):
            return answer

        answer = f"Пациент с id: {id_patient + 1} {not_flag} попал в кластер: {cluster} " \
                 f"потому, что соотвествующие ему значения значимых признаков для кластера {cluster} {not_flag} " \
                 f"похожи на средние или наиболее характерные значения " \
                 f"тех же признаков для пациентов из кластера {cluster}:\n\n"
        sf = interpretation_df.get_significant_features()
        for feature in sf[cluster]:
            if feature in continuous:
                answer += feature + ": пациент (" + str(float(data.iloc[[id_patient]][feature])) + ") / "
                answer += " кластер (" + str(np.mean(data[data[clusters[0]] == cluster][feature])) + ")\n"
            if feature in categorical:
                answer += feature + ": пациент (" + str(data.iloc[[id_patient]][feature].values[0]) + ") / "
                answer += " кластер (" + str(data[data[clusters[0]] == cluster][feature].mode()[0]) + ")\n"

        return answer


@app.callback(
    Output('plots', 'children'),
    Input('set_mode', 'value'),
    [
        State('continuous', 'value'),
        State('categorical', 'value'),
        State('clusters', 'value')
    ]
)
def update_graphs(mode, continuous, categorical, clusters):
    if mode == "None":
        return

    df = data.copy()
    clusters = clusters[-1]
    if mode == "all":
        avg_plots = create_avg_plots(df, continuous, categorical, clusters)
        children = \
            [html.Label(f"Результаты анализа показали, что {df[clusters].count()} пациентов "
                        f"были разделены на {df[clusters].nunique()} группы следующим образом:"),
             dcc.Graph(id="pca_plot", figure=pca_plot(df, clusters)),
             dcc.Graph(id="clsr_plot",
                       figure=px.histogram(df, x=clusters, category_orders=dict(clustert=df[clusters].unique()),
                                           nbins=len(df[clusters].unique()) * 2, color=clusters, text_auto=True))] + \
            [dcc.Graph(id="plot" + str(i), figure=fig) for i, fig in enumerate(avg_plots)]

        return children
    return cluster_info(mode, df, continuous, categorical, clusters)


@app.callback(
    Output('significant_features_table', 'children'),
    Input('submit_button', 'n_clicks'),
    [
        State('continuous', 'value'),
        State('categorical', 'value'),
        State('clusters', 'value'),
        State('dependent', 'value')
    ]
)
def update_significant_features_table(n_clicks, continuous, categorical, clusters, dependent):
    if n_clicks > 0:
        clusters = clusters[-1]
        global interpretation_df

        if interpretation_df is None:
            create_interpretation_df(continuous, categorical, clusters, dependent)

        sf = interpretation_df.get_significant_features()
        sf = pd.DataFrame.from_dict({"Кластеры": list(sf.keys()),
                                     "Значимые признаки": [", ".join(line) for line in list(sf.values())]})

        children = [
            dash_table.DataTable(
                data=sf.to_dict("records"),
                columns=[{"name": i, "id": i} for i in sf.columns],
                page_size=len(sf.index),
                style_table={"margin-top": "10px"},
                style_cell={
                    'minWidth': 95,
                    'maxWidth': 95,
                    'width': 95,
                    'textAlign': 'center'
                },
            ),
        ]

        return children


@app.callback(
    Output("btn_pdf", "disabled"),
    [
        Input('submit_button', 'n_clicks'),
        Input("btn_pdf", "disabled")
    ],
    [
        State('continuous', 'value'),
        State('categorical', 'value'),
        State('clusters', 'value'),
        State('dependent', 'value')
    ]
)
def update_download_button(n_clicks, disabled, continuous, categorical, clusters, dependent):
    if n_clicks > 0:
        clusters = clusters[-1]
        table_lines = [["Clusters", "Significant features"]]

        global interpretation_df
        if interpretation_df is None:
            create_interpretation_df(continuous, categorical, clusters, dependent)

        sf = interpretation_df.get_significant_features()
        for cluster in sf.keys():
            table_lines.append([str(cluster)] + [", ".join(sf[cluster])])

        df = data.rename({clusters: "Clusters"}, axis=1)
        df["Clusters"] = df["Clusters"].astype('str')

        interpretation_report.set_table_lines(table_lines)
        interpretation_report.set_data(df)
        interpretation_report.set_features(continuous=continuous, categorical=categorical)
        interpretation_report.set_differences(interpretation_df.get_differences())
        interpretation_report.create_report()
        disabled = False

    return disabled


@app.callback(
    Output("download_report_pdf", "data"),
    [
        Input("btn_pdf", "n_clicks"),
        Input("btn_pdf", "disabled")
    ]
)
def download_file(n_clicks, disabled):
    if n_clicks > 0 and not disabled:
        return dcc.send_bytes(buffer.getvalue(), filename="report.pdf")


if __name__ == "__main__":
    app.run_server(debug=True)
