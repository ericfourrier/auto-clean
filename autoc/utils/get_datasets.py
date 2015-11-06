# Import Packages
import requests
from lxml import html
import re


# Constant Variables
regex_size = re.compile(r"\(([a-z 0-9\.]+)\)")
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.80 Safari/537.36'
headers = {'user_agent': user_agent, 'X-Requested-With': 'XMLHttpRequest'}
base_url = 'https://www.kaggle.com'


def get_last(l, default=''):
    """ pop from list and return default if empty list  """
    return l.pop() if len(l) else default


def get_competitions_name(headers=headers):
    """ Get all the competitions name from kaggle """
    payload = {'Query': None, 'RewardColumnSort': 'Descending', 'SearchVisibility': 'AllCompetitions',
               'ShowActive': True, 'ShowCompleted': True, 'ShowProspect': True,
               'ShowOpenToAll': True, 'ShowPrivate': False, 'ShowLimited': False}
    url = 'https://www.kaggle.com/competitions/search'

    competition_page = requests.get(url, params=payload, headers=headers)
    tree = html.fromstring(competition_page.text)
    list_competitions = tree.xpath('//table[@id="competitions-table"]//tr//td[1]/a/@href')
    # keep only clean competitions
    competitions_name = [s.replace('/c/', '')for s in list_competitions if s.startswith('/c/')]
    return competitions_name


def generate_url(name):
    """ Generate url of competition from competition name """
    return 'https://www.kaggle.com/c/{}/data'.format(name)

def generate_urls(competitions_name=competitions_name):
    """ Generate urls containing data from competition names """
    return [generate_url(name) for name in competitions_name]


def get_dataset_url(name):
    """ Get a dataset url with some basic infos from a name """
    url = generate_url(name)
    page = requests.get(url, headers=headers)
    tree = html.fromstring(page.text)
    rows = tree.xpath('//table[@id="data-files"]//tbody')
    list_datasets = []
    for row in rows:
        filename = get_last(row.xpath('.//td[@class="file-name"]/text()'))
        for link in row.xpath('.//td[2]//a'):
            dataset = {}
            dataset['competition_name'] = name
            dataset['filename'] = filename
            dataset['url'] = base_url + get_last(link.xpath('./@href'))
            dataset['name'] = get_last(link.xpath('./@name'))
            dataset['size'] = regex_size.search(get_last(link.xpath('./text()'))).group(1)
            list_datasets.append(dataset)
    return list_datasets

def get_all_datasets(random_delay=None):
    """ Get all infos about kaggle datasets """
    list_total_datasets = []
    competition_names = get_competitions_name()
    for name in competition_names:
        try:
            list_total_datasets += get_dataset_url(name)
            print('Datasets from {} gathered'.format(name))
            if random_delay is not None and isinstance(random_delay,int):
                time.sleep(random_delay)
        except Exception as e:
            # bafd but quick
            print(str(e))
            continue
    return list_total_datasets

    
