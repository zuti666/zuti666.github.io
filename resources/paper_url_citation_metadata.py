import time

import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

def fetch_paper_id_from_semantic_scholar(title):
    """
    Fetch the paper ID from Semantic Scholar API using the paper title.

    Args:
        title (str): The title of the paper.

    Returns:
        dict: A dictionary with 'paperId' and 'url' if found, otherwise raises an exception.
    """
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": title,
        "fields": "title,url",
        "limit": 1  # We only need the top result
    }

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get("data"):
            return {
                "paperId": data["data"][0]["paperId"],
                "url": data["data"][0]["url"]
            }
        else:
            raise ValueError(f"No results found for title: {title}")
    else:
        raise ValueError(f"Failed to fetch data from Semantic Scholar. Status code: {response.status_code}")

def fetch_paper_metadata(paper_url, retries=3, delay=2):
    """
    Fetch metadata from a Semantic Scholar paper detail page.

    Args:
        paper_url (str): The Semantic Scholar paper detail URL.

    Returns:
        dict: A dictionary containing the source and PDF URL.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for attempt in range(retries):
        response = requests.get(paper_url, headers=headers)

        # Handle 200 response
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract metadata from <meta> tags
            metadata = {}
            meta_tags = soup.find_all('meta')
            for tag in meta_tags:
                if tag.get('name') == 'citation_pdf_url':
                    metadata['pdf_url'] = tag.get('content')
                if tag.get('name') == 'citation_journal_title':
                    metadata['source'] = tag.get('content')

            return metadata

        # Handle 202 or other temporary statuses
        elif response.status_code == 202:
            print(f"Received 202 status, retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
            time.sleep(delay)
        else:
            print(
                f"Unexpected status code {response.status_code}, retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
            time.sleep(delay)

        # If all retries fail, raise an error
    raise ValueError(f"Failed to fetch metadata from {paper_url}. Final status code: {response.status_code}")


def create_dynamic_badge_with_source(title, output_file):
    """
    Generate a markdown badge for a paper with source information and save it to a file.

    Args:
        title (str): The title of the paper.
        output_file (str): The output markdown file name.
    """
    # Fetch paper ID and URL
    # Fetch paper ID and URL
    try:
        paper_data = fetch_paper_id_from_semantic_scholar(title)
        paper_id = paper_data["paperId"]
        paper_url = paper_data["url"]
        print(f"Paper URL fetched: {paper_url}")
    except ValueError as e:
        print(f"Error fetching paper data: {e}")
        paper_id = None
        paper_url = None



    if paper_id:
        # Define API URL for citation count
        api_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=citationCount"
        encoded_url = quote(api_url, safe='')

        # Construct the badge URL
        badge_url = (
            f"https://img.shields.io/badge/dynamic/json?"
            f"label=citation&query=$.citationCount&url={encoded_url}"
        )
    else:
        badge_url = None

    pdf_url = paper_url
    is_arxiv = None
    if paper_url:
        # Fetch metadata from the paper URL
        try:
            paper_metadata = fetch_paper_metadata(paper_url)
            # print(paper_metadata)
            pdf_url = paper_metadata.get('pdf_url', paper_url)  # Default to paper_url if pdf_url is not found
            print(f'pdf_url: {pdf_url}')
            source = paper_metadata.get('source', '')

            # Determine if it's from arXiv
            is_arxiv = source.lower() == 'arxiv.org' or 'arxiv.org' in pdf_url
            print(f'is_arxiv: {is_arxiv}')
        except ValueError as e:
            print(e)


    # Create markdown content
    markdown_content = (
        f"- **{title}**  \n\n"
        f"   [`Paper`]({pdf_url})  {'`arXiv`' if is_arxiv else ''}   ![citation]({badge_url})\n"

    )

    # Write to the output markdown file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    print('save success')
    return markdown_content

# Example usage
paper_title = "Large Language Models for Information Retrieval: A Survey"
output_filename = "paper_badge_with_metadata.md"

try:
    badge_content = create_dynamic_badge_with_source(paper_title, output_filename)
    print(f'code success end')
except ValueError as e:
    str(e)
