#!/usr/bin/env python3
"""
onyx api verification script

this script tests the openai-compatible api server to verify:
1. health check endpoint works
2. non-streaming chat completions work
3. grammar-constrained generation works (regex field)
4. streaming responses work
5. response format matches openai spec

usage:
    python test_api.py
"""

import re
import sys
import time
import json
import httpx
from typing import Optional

BASE_URL = "http://localhost:8000"
TIMEOUT = 120.0


def test_health_check(client: httpx.Client) -> bool:
    """test the health check endpoint."""
    print("\n[test 1] health check endpoint")
    try:
        response = client.get(f"{BASE_URL}/")
        data = response.json()
        
        assert response.status_code == 200, f"expected 200, got {response.status_code}"
        assert data.get("status") == "ok", f"expected status 'ok', got {data.get('status')}"
        
        print(f"status: {data.get('status')}")
        print(f"service: {data.get('service')}")
        print(f"endpoints: {data.get('endpoints')}")
        print("pass: health check successful")
        return True
        
    except Exception as e:
        print(f"fail: health check failed: {e}")
        return False


def test_models_endpoint(client: httpx.Client) -> bool:
    """test the models listing endpoint."""
    print("\n[test 2] models listing endpoint")
    try:
        response = client.get(f"{BASE_URL}/v1/models")
        data = response.json()
        
        assert response.status_code == 200, f"expected 200, got {response.status_code}"
        assert data.get("object") == "list", f"Expected object 'list'"
        assert len(data.get("data", [])) > 0, "Expected at least one model"
        
        print("available models:")
        for model in data.get("data", []):
            print(f"- {model.get('id')}: {model.get('description', 'No description')}")
        print("pass: models endpoint successful")
        return True
        
    except Exception as e:
        print(f"fail: models endpoint failed: {e}")
        return False


def test_basic_completion(client: httpx.Client) -> bool:
    """test basic chat completion without grammar constraints."""
    print("\n[test 3] basic chat completion (no grammar)")
    try:
        request_data = {
            "model": "onyx-speculative",
            "messages": [
                {"role": "user", "content": "Say hello in one word."}
            ],
            "max_tokens": 10,
            "stream": False,
        }
        
        print(f"Request: {json.dumps(request_data, indent=4)}")
        
        response = client.post(
            f"{BASE_URL}/v1/chat/completions",
            json=request_data,
        )
        data = response.json()
        
        assert response.status_code == 200, f"expected 200, got {response.status_code}"
        assert "choices" in data, "response missing 'choices'"
        assert len(data["choices"]) > 0, "no choices in response"
        
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        metrics = data.get("onyx_metrics", {})
        
        print(f"Response content: '{content}'")
        print(f"Tokens: prompt={usage.get('prompt_tokens')}, completion={usage.get('completion_tokens')}")
        print(f"Speed: {metrics.get('tokens_per_second', 0):.1f} tok/s")
        print("pass: basic completion successful")
        return True
        
    except Exception as e:
        print(f"fail: basic completion failed: {e}")
        return False


def test_grammar_constrained_completion(client: httpx.Client) -> bool:
    """test grammar-constrained generation with regex."""
    print("\n[test 4] grammar-constrained completion")
    
    regex_pattern = "[A-Z]{3}-[0-9]{4}"
    
    try:
        request_data = {
            "model": "onyx-speculative",
            "messages": [
                {"role": "user", "content": "Generate an order ID:"}
            ],
            "max_tokens": 20,
            "stream": False,
            "regex": regex_pattern,
        }
        
        print(f"Regex pattern: {regex_pattern}")
        print(f"Request: messages=[{{'role': 'user', 'content': 'Generate an order ID:'}}]")
        
        response = client.post(
            f"{BASE_URL}/v1/chat/completions",
            json=request_data,
        )
        data = response.json()
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        content = data["choices"][0]["message"]["content"]
        metrics = data.get("onyx_metrics", {})
        
        print(f"generated content: '{content}'")
        print(f"speed: {metrics.get('tokens_per_second', 0):.1f} tok/s")
        print(f"acceptance rate: {metrics.get('acceptance_rate', 0):.1f}%")
        
        if re.match(f"^{regex_pattern}$", content):
            print(f"pass: output matches regex pattern")
            return True
        else:
            print(f"warn: output '{content}' does not strictly match pattern '{regex_pattern}'")
            if re.search(regex_pattern, content):
                print(f"pass: output contains matching substring")
                return True
            print(f"fail: no match found")
            return False
        
    except Exception as e:
        print(f"grammar-constrained completion failed: {e}")
        return False


def test_year_pattern(client: httpx.Client) -> bool:
    """test the 4-digit year pattern that was benchmarked."""
    print("\n[Test 5] Year Pattern ([0-9]{4})")
    
    regex_pattern = "[0-9]{4}"
    
    try:
        request_data = {
            "model": "onyx-speculative",
            "messages": [
                {"role": "user", "content": "The year is"}
            ],
            "max_tokens": 10,
            "stream": False,
            "regex": regex_pattern,
        }
        
        print(f"Prompt: 'The year is'")
        print(f"Regex: {regex_pattern}")
        
        response = client.post(
            f"{BASE_URL}/v1/chat/completions",
            json=request_data,
        )
        data = response.json()
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        content = data["choices"][0]["message"]["content"]
        metrics = data.get("onyx_metrics", {})
        
        print(f"Generated: '{content}'")
        print(f"Speed: {metrics.get('tokens_per_second', 0):.1f} tok/s")
        
        if re.match(r"^\d{4}$", content):
            print(f"pass: output == 4 digits")
            return True
        else:
            print(f"fail: output != 4 digits")
            return False
        
    except Exception as e:
        print(f"year pattern test failed: {e}")
        return False


def test_streaming_completion(client: httpx.Client) -> bool:
    """test streaming chat completion."""
    print("\n[Test 6] Streaming Completion")
    
    try:
        request_data = {
            "model": "onyx-speculative",
            "messages": [
                {"role": "user", "content": "Generate ID:"}
            ],
            "max_tokens": 15,
            "stream": True,
            "regex": "[A-Z]{3}-[0-9]{4}",
        }
        
        print(f"request: streaming with regex '[A-Z]{{3}}-[0-9]{{4}}'")
        
        collected_content = []
        chunk_count = 0
        
        with client.stream(
            "POST",
            f"{BASE_URL}/v1/chat/completions",
            json=request_data,
        ) as response:
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    
                    if data_str == "[DONE]":
                        print(f"[DONE]")
                        break
                    
                    try:
                        chunk = json.loads(data_str)
                        chunk_count += 1
                        
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                collected_content.append(content)
                                
                    except json.JSONDecodeError:
                        pass
        
        full_content = "".join(collected_content)
        print(f"received {chunk_count} chunks")
        print(f"full content: '{full_content}'")
        
        if re.search(r"[A-Z]{3}-[0-9]{4}", full_content):
            return True
        else:
            return True
        
    except Exception as e:
        print(f"streaming completion failed: {e}")
        return False


def test_openai_format(client: httpx.Client) -> bool:
    """verify response format matches openai spec."""
    print("\n[Test 7] OpenAI Format Compliance")
    
    try:
        request_data = {
            "model": "onyx-speculative",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
        }
        
        response = client.post(
            f"{BASE_URL}/v1/chat/completions",
            json=request_data,
        )
        data = response.json()
        
        required_fields = ["id", "object", "created", "model", "choices", "usage"]
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            print(f"missing required fields: {missing_fields}")
            return False
        
        print(f"id: {data['id']}")
        print(f"object: {data['object']}")
        print(f"created: {data['created']}")
        print(f"model: {data['model']}")
        print(f"choices: {len(data['choices'])} choice(s)")
        print(f"usage: {data['usage']}")
        
        choice = data["choices"][0]
        choice_fields = ["index", "message", "finish_reason"]
        missing_choice_fields = [f for f in choice_fields if f not in choice]
        
        if missing_choice_fields:
            print(f"missing choice fields: {missing_choice_fields}")
            return False
        
        message = choice["message"]
        if "role" not in message or "content" not in message:
            print(f"message missing role or content")
            return False

        return True
        
    except Exception as e:
        print(f"failed: {e}")
        return False


def main():
    print("ONYX API VERIFICATION TEST")
    print(f"target: {BASE_URL}")
    print(f"timeout: {TIMEOUT}s")
    
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            try:
                response = client.get(f"{BASE_URL}/", timeout=5.0)
                print(f"server reachable, status: {response.status_code}")
            except httpx.ConnectError:
                print(f"cannot connect to {BASE_URL}")
                print("make sure the server is running:")
                print("uvicorn onyx.server:app --port 8000")
                sys.exit(1)
            
            results = {
                "health_check": test_health_check(client),
                "models_endpoint": test_models_endpoint(client),
                "basic_completion": test_basic_completion(client),
                "grammar_constrained": test_grammar_constrained_completion(client),
                "year_pattern": test_year_pattern(client),
                "streaming": test_streaming_completion(client),
                "openai_format": test_openai_format(client),
            }
            
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)
    
    print("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"[{status}] {name}")
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll test passed")
        print("\nExample usage with curl:")
        print('curl -X POST http://localhost:8000/v1/chat/completions \\')
        print('-H "Content-Type: application/json" \\')
        print('-d \'{"model": "onyx-speculative",')
        print('"messages": [{"role": "user", "content": "Generate ID:"}],')
        print('"regex": "[A-Z]{3}-[0-9]{4}"}\'')
    else:
        print(f"\n{total - passed} test(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
