import json
from typing import List, Dict, Any, Union


def extract_content_and_urls(value: Dict[str, Any]) -> List[Dict[str, Union[str, Dict[str, str]]]]:
    result = []
    possible_keys = ['call_tool', 'date_finder', 'embedding_retriever']

    for key in possible_keys:
        if key in value:
            data = value[key]
            if 'messages' in data:
                messages = data['messages']
                if isinstance(messages, list) and len(messages) > 0:
                    message = messages[0]
                    content = message.content
                    # Check if the content is a JSON string
                    try:
                        json_content = json.loads(content)
                        # Handle case where content is a JSON string
                        for item in json_content:
                            url = item.get('url')
                            content = item.get('content')
                            result.append({'url': url, 'content': content})
                    except json.JSONDecodeError:
                        # Handle case where content is a regular string
                        result.append({'content': content})
            break  # Stop after finding the first valid key
    return result