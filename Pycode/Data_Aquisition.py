import pandas as pd
import os
import json
import requests



def get_crypto_data(n_batch_obs=8,  currency='EUR', exchange='Coinbase'):
    '''
    GET DATA from CryptoCompare API
    '''
    url_call = 'https://min-api.cryptocompare.com/data/v2/histohour?fsym=BTC&tsym={}&limit={}&e={}'

    # get data
    reqs = []


    # Initial Call
    req = json.loads(requests.get(url_call.format(currency, 2000, exchange)).text)
    print('- {}/BTC Prices from {} to {} '.format(currency, req['Data']['TimeFrom'], req['Data']['TimeTo']))
    reqs.append(req)

    for i in range(1, n_batch_obs):
        # Second query to ave double the history if n_obs > 2000
        req = json.loads(
            requests.get(url_call.format(currency, 2000, exchange) + '&toTs={}'.format(req['Data']['TimeFrom'])).text)
        print('- {}/BTC Prices from {} to {} '.format(currency, req['Data']['TimeFrom'], req['Data']['TimeTo']))
        reqs.append(req)


    # Format as dataframe & sort DatetimeIndex
    df = pd.concat([pd.DataFrame(req['Data']['Data']) for req in reqs], axis=0)
    df.index = pd.to_datetime(df['time'], origin='unix', unit='s')
    df.drop(columns=['time', 'conversionType', 'conversionSymbol'], inplace=True)
    df.sort_index(ascending=True, inplace=True)
    df.drop_duplicates(inplace=True)

    df.info()

    # Format as dataframe & sort DatetimeIndex
    df = pd.concat([pd.DataFrame(req['Data']['Data']) for req in reqs], axis=0)
    df.index = pd.to_datetime(df['time'], origin='unix', unit='s')
    df.drop(columns=['time', 'conversionType', 'conversionSymbol'], inplace=True)
    df.sort_index(ascending=True, inplace=True)
    df.drop_duplicates(inplace=True)

    print(df.info())

    return df


def get_news():
    '''
    GET News text data from the CyrptoCompare API
    '''

    # Main URL
    url_news = 'https://min-api.cryptocompare.com/data/v2/news/?lang=EN&excludeCategories=ETH,LTC,XMR,ZEC,XRP,TRX,ADA,DASH,XTZ,USDT&feeds=coindesk,yahoofinance,cointelegraph,bitcoin.com&ITs=1486506200'

    with open(os.path.join(os.path.abspath('../../'), 'creds.txt'), 'r') as credentials:
        api_key = credentials.read().split(':')[-1]

    req = json.loads(requests.get(url_news + api_key).text)

    df_news = pd.DataFrame(req['Data'])
    df_news.index = pd.to_datetime(df_news['published_on'], origin='unix', unit='s')
    df_news.drop(columns=['published_on', 'id', 'guid', 'imageurl', 'url', 'source', 'upvotes', 'downvotes', 'lang',
                          'source_info'], inplace=True)
    df_news.sort_index(ascending=True, inplace=True)

    return df_news




if __name__ == '__main__':
    get_crypto_data(n_batch_obs=8, currency='EUR', exchange='Coinbase')