from dash_bootstrap_components._components.Col import Col
import dash
from dash_bootstrap_components._components.Card import Card
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn import linear_model
import os


#AÑADIENDO RUTA Y ABRIENDO ARCHIVO CON PANDAS 
path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path, "misc", "cars.csv")

df = pd.read_csv(path)


# CONSTRUYENDO TABLA DE DATOS
def draw_table():

    layout=go.Layout(
        showlegend = False,
        #title='Tabla carros',
        #template='plotly_dark',
        #plot_bgcolor= 'rgba(0, 0, 0, 0)',
        paper_bgcolor= 'rgba(0, 0, 0, 0)',
        margin=dict(
            r=10, l=10,
            b=10, t=10)
     )

    config = {'displayModeBar': False}

    # DIBUJANDO TABLA
    table = go.Figure(layout=layout)
    table.add_trace(
        go.Table(
            header = dict(values=list(df.columns),
            line_color='black',
            fill_color='paleturquoise',
            align='center',
            font_color='black'
            ),
            cells = dict(
                values=[df.Marca, df.Modelo, df.Volumen, df.Peso, df.CO2],
                line_color='black',
                fill_color='lavender',
                align='left',
                font_color='black'
            )
        )
            
    )

    table_show = html.Div([
                    dbc.Card(
                        dbc.CardBody([
                            html.H1("Tabla de datos"),
                            dcc.Graph(
                                figure = table, config = config
                            )
                        ])
                    )
                ])


    return table_show
    

# CONSTRUYENDO TABLA DE ANÁLISIS DESCRIPTIVO
def analisis_descriptivo():

    prom = [
        round(df['Volumen'].mean()),
        round(df['Peso'].mean()),
        round(df['CO2'].mean())
        ]
    mediana = [
        round(df['Volumen'].median()),
        round(df['Peso'].median()),
        round(df['CO2'].median())
        ]
    moda = [
        round(df['Volumen'].mode()),
        round(df['Peso'].mode()),
        round(df['CO2'].mode())
        ]
    max = [
        round(df['Volumen'].max()),
        round(df['Peso'].max()),
        round(df['CO2'].max())
        ]
    min = [
        round(df['Volumen'].min()),
        round(df['Peso'].min()),
        round(df['CO2'].min())
        ]
    std = [
        round(df['Volumen'].std()),
        round(df['Peso'].std()),
        round(df['CO2'].std())
        ]
    cant = [
        round(df['Volumen'].count()),
        round(df['Peso'].count()),
        round(df['CO2'].count())
        ]

    table_header = [
        html.Thead(
            html.Tr([
                html.Th('Estadístico'),
                html.Th('Volumen'),
                html.Th('Peso'),
                html.Th('CO2')
            ])
        )
    ]
    row0 = html.Tr([html.Td('Cantidad de datos'), 
                    html.Td(cant[0]),
                    html.Td(cant[1]),
                    html.Td(cant[2])
            ])
    row1 = html.Tr([html.Td('Promedio'), 
                    html.Td(prom[0]),
                    html.Td(prom[1]),
                    html.Td(prom[2])
            ])
    row2 = html.Tr([html.Td('Mediana'), 
                    html.Td(mediana[0]),
                    html.Td(mediana[1]),
                    html.Td(mediana[2])
            ])
    row3 = html.Tr([html.Td('Moda'), 
                    html.Td(moda[0]),
                    html.Td(moda[1]),
                    html.Td(moda[2])
            ])
    row4 = html.Tr([html.Td('Valor máximo'), 
                    html.Td(max[0]),
                    html.Td(max[1]),
                    html.Td(max[2])
            ])
    row5 = html.Tr([html.Td('Valor mínimo'), 
                    html.Td(min[0]),
                    html.Td(min[1]),
                    html.Td(min[2])
            ])
    row6 = html.Tr([html.Td('Desviación estandar'), 
                    html.Td(std[0]),
                    html.Td(std[1]),
                    html.Td(std[2])
            ])


    table_body = [html.Tbody([row0, row1, row2, row3, row4, row5, row6])]

    table_est_desc = html.Div([
                        dbc.Card(
                            dbc.CardBody([
                                html.H1("Análisis descriptivo"),
                                html.Br(),
                                html.Br(),
                                dbc.Table(table_header + table_body, bordered = True)
                            ])
                        )
                    ])

    return table_est_desc


# CONSTRUYENDO RADIO ITEMS DE GRÁFICO DE BARRAS
def radio_items_barras():

    cabeceras = list(df.columns)
    radio_items = html.Div([

                        html.H1("Gráfico de barras"),
                        dcc.RadioItems(
                            id='radio_ejey_barras',
                            options=[{'label': i, 'value': i} for i in cabeceras[2:]],
                            value= cabeceras[2],
                            labelStyle={
                                'display': 'inline-block',
                                'paddingRight': '20px'
                                }
                        )


                  ])

    return radio_items

# CONTRUYENDO GRAFICO DE BARRAS CON CALLBACKS
def grafico_barras():

    @app.callback(
        Output('grafico_barras', 'figure'),
        Input('radio_ejey_barras', 'value')
    )   
        # DIBUJANDO GRAFICO DE BARRAS
    def act_grafico_barras(radio_ejey_barras):

        layout=go.Layout(

            showlegend = False,
            paper_bgcolor= 'rgba(0, 0, 0, 0)',
            yaxis_color = 'white',
            xaxis_color = 'white',
            xaxis={'title':'Modelos de autos'},
            yaxis={'title': radio_ejey_barras},
            margin=dict(
            r=10, l=10,
            b=10, t=10)

        )
        
        # DIBUJANDO GRAFICO DE BARRAS
        fig_barras = go.Figure(layout=layout)
        fig_barras.add_trace(

            go.Bar(
                y= df[radio_ejey_barras],
                x= df['Modelo'],
                text = ["Marca: {}".format(u) for u in df['Marca']]

            )

        )
        return fig_barras

    # DIBUJANDO MARCO DEL GRAFICO DE BARRAS    
    config = {'displayModeBar': False}
    barra = html.Div([
                        dbc.Card(
                            dbc.CardBody([
                                radio_items_barras(),
                                dcc.Graph(id='grafico_barras', config = config)
                            ])
                        )
                  ])
    
    return barra



# GRÁFICO DE DISPERSIÓN
def grafico_dispersion(variable):

    layout=go.Layout(

            showlegend = False,
            paper_bgcolor= 'rgba(0, 0, 0, 0)',
            yaxis_color = 'white',
            xaxis_color = 'white',
            xaxis={'title':variable},
            yaxis={'title': 'CO2'},
            margin=dict(
            r=10, l=10,
            b=10, t=10)

        )

    fig_dispersion = go.Figure(layout=layout)
    fig_dispersion.add_trace(

            go.Scatter(

                y= df['CO2'],
                x= df[variable],
                mode='markers',
                text = [
                        '''Marca: {} <br>Modelo: {}'''.format(Ma,Mo) 
                        for Ma in df['Marca'] for Mo in df['Modelo']
                    ]
            )

        )

    # DIBUJANDO MARCOS DE LOS GRÁFICOS DE DISPERSIÓN
    config = {'displayModeBar': False}
    dispersion = html.Div([
                        dbc.Card(
                            dbc.CardBody([
                                html.H1("Gráfico de dispersión - {}".format(variable)),
                                dcc.Graph(figure = fig_dispersion, config = config)
                            ])
                        )
                  ])
    
    return dispersion

# ANÁLISIS DE REGRESIÓN

def regresion_multiple():

    X = df[['Peso', 'Volumen']]
    Y = df['CO2']

    regresion = linear_model.LinearRegression()
    regresion.fit(X,Y)

    r2 = regresion.score(X,Y)
    intercepto = regresion.intercept_
    coef= regresion.coef_
    coef_r = [round(coef[0],8), round(coef[1],8)]

    y_pred = regresion.predict(X)

    layout=go.Layout(

            showlegend = False,
            paper_bgcolor= 'rgba(0, 0, 0, 0)',
            yaxis_color = 'white',
            xaxis_color = 'white',
            xaxis={'title':'Peso y Volumen'},
            yaxis={'title': 'CO2'}

        )

    # DIBUAJNDO GRÁFICO DE DISPERSIÓN
    fig_regresion = go.Figure(layout=layout)
    fig_regresion.add_trace(

        go.Scatter3d(

            z= df['CO2'],
            x= df['Peso'],
            y= df['Volumen'],
            mode='markers',
            text = [
                    '''Marca: {} <br>Modelo: {}'''.format(Ma,Mo) 
                    for Ma in df['Marca'] for Mo in df['Modelo']
                ]

        )

    )

    fig_regresion.add_trace(

        go.Mesh3d(
            z= df['CO2'],
            x= df['Peso'],
            y= df['Volumen'],
            color='rgba(244,22,100,0.6)',
            opacity=0.5,
            text = [
                    '''Marca: {} <br>Modelo: {}'''.format(Ma,Mo) 
                    for Ma in df['Marca'] for Mo in df['Modelo']
                ]

        )

    )

    fig_regresion.update_layout(scene = dict(
                    xaxis = dict(

                         tickfont=dict(color='white'),
                         backgroundcolor="rgb(200, 200, 230)",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",
                         color = 'white',
                         title='Peso'
                          ),
                         

                    yaxis = dict(

                        tickfont=dict(color='white'),
                        backgroundcolor="rgb(230, 200,230)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",
                        color= 'white',
                        title='Volumen'
                        ),

                    zaxis = dict(

                        tickfont=dict(color='white'),
                        backgroundcolor="rgb(230, 230,200)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",
                        color= 'white',
                        title='CO2',
                        ),
                    ),

                    width=1000,
                    margin=dict(
                    r=10, l=10,
                    b=10, t=10)
                  )

    parrafo1 = "R2 = {}, coeficientes = {} y el intercepto = {} ".format(round(r2,2), coef, round(intercepto,5))
    parrafo2 = "Ecuación: Y = {}X1 + {}x2 + {}".format(coef_r[0], coef_r[1], round(intercepto,2))
    parrafo3 = '''Estos valores nos dicen que si el peso aumenta en 1 kg, la emisión de CO2 aumenta en {}g.
                Y si el tamaño del motor (Volumen) aumenta en 1 cm3 , la emisión de CO2 aumenta en {}g.'''.format(coef_r[0], coef_r[1])


    config = {'displayModeBar': False}
    regresion_multiple = html.Div([
                        dbc.Card(
                            dbc.CardBody([
                                html.H1("Gráfico de dispersión - Regresión multiple"),
                                dcc.Graph(figure = fig_regresion, config = config),
                                html.Br(),
                                html.Div([

                                    html.H3('Datos:'),
                                    html.P(parrafo1),
                                    html.P(parrafo2),
                                    html.P(parrafo3)
                                    
                                ])
                                
                            ])
                        )
                  ])
    
    return regresion_multiple

# CONSTRUYENDO ESTIMACIONES
def estimaciones():

    entrada = html.Div([
                dbc.Card(
                    dbc.CardBody([
                        html.H1('Estimaciones'),
                        html.Br(),
                        html.Div([
                            "Peso:    ",
                            dcc.Input(id='input_peso', value=0, type='number',required=True, className='ml-4')
                            ]),
                        html.Br(),
                        html.Div([
                            "Volumen: ",
                            dcc.Input(id='input_vol', value=0, type='number',required=True)
                            ]),
                        html.Br(),
                        html.Button("Estimar", id="btn_pred", className='btn btn-primary'),
                        html.Br(),
                        html.Br(),
                        html.P(id='output_estimacion'),
                        html.H3('Descripción: '),
                        html.P('''
                                    Este es mi primer proyecto de Dashboard, en Python hecho en Dash que utiliza librerías como: 
                                    Plotly, Bootstrap para dash, pandas, numpy y Scikit-learn; El cual puedes encontrar un pequeña
                                    guía de como estructurar un Dashboard, combinando diferentes formas construir y darle estilo
                                    a los distintos elementos. Espero que les sirva de ayuda a los que también están empezando.
                        
                        '''),
                        html.P('''
                                    En el siguiente Dashboard vamos a analizar la emisión de CO2 de un automóvil 
                                    en función del tamaño del motor en Volumen y el peso del automóvil,
                                    para realizar estimaciones.
                        
                        '''),
                        html.P('Realizado por: Jose Noguera | josenoguera@gmail.com', className='mb-4')
                    ])
                )

    ])

    @app.callback(
        Output('output_estimacion', 'children'),
        [Input('btn_pred', 'n_clicks')],
        State('input_peso', 'value'),
        State('input_vol', 'value'),
    )

    def estimar(n_clicks, input_peso, input_vol):
        
            if n_clicks is None:
                resultado = 'Ingrese los valores de Peso y Volumen para la estimación.'
            else:
                X = df[['Peso', 'Volumen']]
                Y = df['CO2']
                regresion = linear_model.LinearRegression()
                regresion.fit(X,Y)
                pred_CO2 = regresion.predict([[input_peso,input_vol]])    
                resultado = 'Para un automóvil cuyo peso es de {} kg y el volumen es de {} cm3, la emisión de CO2 es: {}'.format(input_peso, input_vol, round(pred_CO2[0],3))

            return resultado

    return entrada

# CONFIGURANDO BOOTSTRAP
external_stylesheets = [dbc.themes.SLATE]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# CONSTRUYENDO DASHBOARD
body = html.Div([
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(
                            html.Div([
                                html.H1('Dashboard')
                            ]), width=2
                        )
                    ], align='center', justify='center'),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            draw_table()
                        ]),
                        dbc.Col([
                            analisis_descriptivo()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            grafico_barras()
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            grafico_dispersion('Peso')
                        ]),
                        dbc.Col([
                            grafico_dispersion('Volumen')
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            regresion_multiple()
                        ], width=8),
                        dbc.Col([
                            estimaciones()
                        ], width=4 , className='pl-2' )
                    ])
                ]), color = 'dark'
        )
],className="text-light")

# DIBUJAN EL BODY EN EL LAYOUT DE DASH

app.layout = html.Div([body])


# CORRER SERVIDOR
if __name__ == '__main__':
    app.run_server(debug=True)