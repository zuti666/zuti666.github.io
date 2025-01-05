import asyncio
import json
import requests
from pyppeteer import launch
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

async def fetch_paper_metadata_with_pyppeteer(paper_url):
    """
    Fetch metadata from a Semantic Scholar paper detail page using pyppeteer.

    Args:
        paper_url (str): The Semantic Scholar paper detail URL.

    Returns:
        dict: A dictionary containing metadata like source, PDF URL, journal title, and publication date.
    """
    browser = await launch(headless=False, slowMo=50)  # Show browser, slowMo adds delay between actions
    try:
        page = await browser.newPage()
        await page.goto(paper_url, waitUntil='networkidle2')
        await asyncio.sleep(1)  # Wait for dynamic content to load

        # Handle cookie consent
        try:
            # Look for the "Accept All Cookies" button and click it
            await page.waitForSelector('#app > div.cookie-banner > div > div.cookie-banner__actions > button',
                                       timeout=1000)
            await page.click('#app > div.cookie-banner > div > div.cookie-banner__actions > button')
            print("Cookies accepted.")
        except Exception as e:
            print(f"No cookie consent button found or clickable: {e}")


        # Extract page content
        content = await page.content()
        print("Page content loaded successfully.")

        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')

        metadata = {}
        # Extract metadata from <meta> tags
        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            if tag.get('name') == 'citation_pdf_url':
                metadata['pdf_url'] = tag.get('content')
            if tag.get('name') == 'citation_journal_title':
                metadata['journal_title'] = tag.get('content')
            if tag.get('name') == 'citation_publication_date':
                metadata['publication_date'] = tag.get('content')

        # Extract JSON-LD data for additional metadata
        script_tag = soup.find('script', {'type': 'application/ld+json', 'class': 'schema-data'})
        if script_tag:
            json_data = json.loads(script_tag.string)
            graph_data = json_data.get('@graph', [])
            if isinstance(graph_data, list):
                for item in graph_data:
                    if isinstance(item, list): # 进入第三个元素，列表
                        for sub_item in item:  # 便利李彪
                            print(f'subitem: {sub_item}')
                            if isinstance(sub_item, dict) and sub_item.get('@type') == 'Article': #
                                metadata = {
                                    'pdf_url': sub_item.get('mainEntity', ''),
                                    'publication_date': sub_item.get('datePublished', ''),
                                    'headline': sub_item.get('headline', ''),
                                    # 'authors': [author.get('name') for author in sub_item.get('author', []) if
                                    #             author.get('name')],
                                    'publisher_name': sub_item.get('publisher', {}).get('name', ''),
                                }
                                print(f'metadata: {metadata}')
                                break



        return metadata
    finally:
        await browser.close()  # Ensure browser closes even if an error occurs

def fetch_metadata(paper_url):
    """
    Wrapper to run pyppeteer in a synchronous context.

    Args:
        paper_url (str): The Semantic Scholar paper URL.

    Returns:
        dict: Metadata of the paper.
    """
    return asyncio.run(fetch_paper_metadata_with_pyppeteer(paper_url))

def create_dynamic_badge_with_source(title, output_file):
    """
    Generate a markdown badge for a paper with source information and save it to a file.

    Args:
        title (str): The title of the paper.
        output_file (str): The output markdown file name.
    """
    # Fetch paper ID and URL
    try:
        paper_data = fetch_paper_id_from_semantic_scholar(title)
        paper_id = paper_data["paperId"]
        paper_url = paper_data["url"]
        print(f"Paper URL fetched: {paper_url}")
    except ValueError as e:
        print(f"Error fetching paper data: {e}")
        return

    # Fetch metadata
    try:
        paper_metadata = fetch_metadata(paper_url)
    except Exception as e:
        print(f"Error fetching metadata with pyppeteer: {e}")
        return

    pdf_url = paper_metadata.get('pdf_url', '')
    journal_title = paper_metadata.get('journal_title', '')
    publisher_name = paper_metadata.get('publisher_name', '')
    publication_date = paper_metadata.get('exact_publication_date', paper_metadata.get('publication_date', ''))
    is_arxiv = 'arxiv.org' in pdf_url.lower() or journal_title.lower() == 'arxiv.org'

    # Define badge URL
    api_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=citationCount"
    encoded_url = quote(api_url, safe="")
    badge_url = (
        f"https://img.shields.io/badge/dynamic/json?"
        f"label=citation&query=citationCount&url={encoded_url}"
    )

    # Create markdown content
    markdown_content = (
        f"- **{title}**\n\n"
        f"  {publication_date}   [`semanticscholar`]({paper_url})  [`Paper`]({pdf_url})  "
        f"{'`arXiv`' if is_arxiv else ''}   {journal_title if journal_title else publisher_name}\n\n"
        f"  ![citation]({badge_url})\n"
    )

    # Write to file
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(markdown_content + '\n\n')
    print("Markdown content saved.")
    return markdown_content

def main():
    paper_title = "Query2doc: Query Expansion with Large Language Models"
    output_filename = "paper_badge_with_metadata.md"

    try:
        badge_content = create_dynamic_badge_with_source(paper_title, output_filename)
        print("Markdown generated successfully.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
