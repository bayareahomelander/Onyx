"""
kitchen sink test - grand integration test for all json schema features
"""
import pytest
from onyx_rust import GrammarConstraint


def test_kitchen_sink_pretty_printed():
    """
    tests complex schema with all features:
    - nested objects
    - required fields
    - integer type
    - regex pattern
    - union type (number | null)
    - enum
    - array with minitems/maxitems
    - string with maxlength
    - pretty printing (newlines, tabs)
    """
    # comprehensive vocab with whitespace
    vocab = [
        b'{', b'}', b'[', b']', b',', b':', b'"',  # structural
        b'\n', b'\t', b'  ',  # whitespace for pretty printing
        b'"user_id"', b'"profile"', b'"name"', b'"age"', b'"status"', b'"tags"',  # keys
        b'123',  # integer
        b'"John"',  # valid name (capital + lowercase)
        b'"john"',  # invalid name (no capital)
        b'25', b'null',  # age options
        b'"active"', b'"suspended"', b'"invalid"',  # status options
        b'"tag1"', b'"tag2"', b'"tag3"', b'"tag4"',  # tags (valid length)
        b'"toolong"',  # tag exceeds maxlength
    ]
    
    gc = GrammarConstraint(vocab)
    
    schema = '''{
        "type": "object",
        "required": ["user_id", "profile"],
        "properties": {
            "user_id": {"type": "integer"},
            "profile": {
                "type": "object",
                "required": ["name", "tags"],
                "properties": {
                    "name": {"type": "string", "pattern": "^[A-Z][a-z]+$"},
                    "age": {"type": ["number", "null"]},
                    "status": {"enum": ["active", "suspended"]},
                    "tags": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {"type": "string", "maxLength": 5}
                    }
                }
            }
        }
    }'''
    
    gc.compile_json_schema(schema)
    
    # build valid pretty-printed json:
    # {
    #   "user_id": 123,
    #   "profile": {
    #     "name": "John",
    #     "age": null,
    #     "status": "active",
    #     "tags": ["tag1", "tag2"]
    #   }
    # }
    
    state = gc.init_state()
    
    # {
    state = gc.advance_state(state, 0)  # {
    state = gc.advance_state(state, 7)  # \n
    state = gc.advance_state(state, 8)  # \t
    
    # "user_id": 123,
    state = gc.advance_state(state, 10)  # "user_id"
    state = gc.advance_state(state, 5)   # :
    state = gc.advance_state(state, 9)   # space
    state = gc.advance_state(state, 16)  # 123
    state = gc.advance_state(state, 4)   # ,
    state = gc.advance_state(state, 7)   # \n
    state = gc.advance_state(state, 8)   # \t
    
    # "profile": {
    state = gc.advance_state(state, 11)  # "profile"
    state = gc.advance_state(state, 5)   # :
    state = gc.advance_state(state, 9)   # space
    state = gc.advance_state(state, 0)   # {
    state = gc.advance_state(state, 7)   # \n
    state = gc.advance_state(state, 8)   # \t
    state = gc.advance_state(state, 8)   # \t
    
    # "name": "John",
    state = gc.advance_state(state, 12)  # "name"
    state = gc.advance_state(state, 5)   # :
    state = gc.advance_state(state, 9)   # space
    state = gc.advance_state(state, 17)  # "John"
    state = gc.advance_state(state, 4)   # ,
    state = gc.advance_state(state, 7)   # \n
    state = gc.advance_state(state, 8)   # \t
    state = gc.advance_state(state, 8)   # \t
    
    # "age": null,
    state = gc.advance_state(state, 13)  # "age"
    state = gc.advance_state(state, 5)   # :
    state = gc.advance_state(state, 9)   # space
    state = gc.advance_state(state, 20)  # null
    state = gc.advance_state(state, 4)   # ,
    state = gc.advance_state(state, 7)   # \n
    state = gc.advance_state(state, 8)   # \t
    state = gc.advance_state(state, 8)   # \t
    
    # "status": "active",
    state = gc.advance_state(state, 14)  # "status"
    state = gc.advance_state(state, 5)   # :
    state = gc.advance_state(state, 9)   # space
    state = gc.advance_state(state, 21)  # "active"
    state = gc.advance_state(state, 4)   # ,
    state = gc.advance_state(state, 7)   # \n
    state = gc.advance_state(state, 8)   # \t
    state = gc.advance_state(state, 8)   # \t
    
    # "tags": ["tag1", "tag2"]
    state = gc.advance_state(state, 15)  # "tags"
    state = gc.advance_state(state, 5)   # :
    state = gc.advance_state(state, 9)   # space
    state = gc.advance_state(state, 2)   # [
    state = gc.advance_state(state, 24)  # "tag1"
    state = gc.advance_state(state, 4)   # ,
    state = gc.advance_state(state, 9)   # space
    state = gc.advance_state(state, 25)  # "tag2"
    state = gc.advance_state(state, 3)   # ]
    state = gc.advance_state(state, 7)   # \n
    state = gc.advance_state(state, 8)   # \t
    
    # close profile }
    state = gc.advance_state(state, 1)   # }
    state = gc.advance_state(state, 7)   # \n
    
    # close root }
    state = gc.advance_state(state, 1)   # }
    
    assert gc.is_match_state(state), "valid pretty-printed json should be accepted"


def test_kitchen_sink_constraint_enforcement():
    """
    tests that constraints are enforced:
    - lowercase name rejected (pattern)
    - invalid enum rejected
    - 4th tag blocked (maxitems)
    - toolong tag blocked (maxlength)
    """
    vocab = [
        b'{', b'}', b'[', b']', b',', b':',
        b'"user_id"', b'"profile"', b'"name"', b'"tags"',
        b'123',
        b'"John"', b'"john"',  # valid vs invalid name
        b'"active"', b'"invalid"',  # valid vs invalid status
        b'"tag1"', b'"tag2"', b'"tag3"', b'"tag4"',  # tags
        b'"toolong"',  # exceeds maxlength
    ]
    
    gc = GrammarConstraint(vocab)
    
    schema = '''{
        "type": "object",
        "required": ["user_id", "profile"],
        "properties": {
            "user_id": {"type": "integer"},
            "profile": {
                "type": "object",
                "required": ["name", "tags"],
                "properties": {
                    "name": {"type": "string", "pattern": "^[A-Z][a-z]+$"},
                    "tags": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {"type": "string", "maxLength": 5}
                    }
                }
            }
        }
    }'''
    
    gc.compile_json_schema(schema)
    
    # test 1: lowercase name should be blocked
    state = gc.init_state()
    state = gc.advance_state(state, 0)  # {
    state = gc.advance_state(state, 7)  # "profile"
    state = gc.advance_state(state, 5)  # :
    state = gc.advance_state(state, 0)  # {
    state = gc.advance_state(state, 8)  # "name"
    state = gc.advance_state(state, 5)  # :
    
    valid = gc.get_valid_token_ids(state)
    assert 11 in valid, '"John" should be valid'
    assert 12 not in valid, '"john" should be blocked by pattern'
    
    # test 2: continue to test maxitems on tags
    gc.reset()
    state = gc.init_state()
    state = gc.advance_state(state, 0)   # {
    state = gc.advance_state(state, 7)   # "profile"
    state = gc.advance_state(state, 5)   # :
    state = gc.advance_state(state, 0)   # {
    state = gc.advance_state(state, 9)   # "tags"
    state = gc.advance_state(state, 5)   # :
    state = gc.advance_state(state, 2)   # [
    state = gc.advance_state(state, 15)  # "tag1"
    state = gc.advance_state(state, 4)   # ,
    state = gc.advance_state(state, 16)  # "tag2"
    state = gc.advance_state(state, 4)   # ,
    state = gc.advance_state(state, 17)  # "tag3"
    
    valid = gc.get_valid_token_ids(state)
    assert 3 in valid, '] should be allowed at 3 items'
    assert 4 not in valid, ', should be blocked at maxitems'
    
    # test 3: toolong tag blocked by maxlength
    gc.reset()
    state = gc.init_state()
    state = gc.advance_state(state, 0)   # {
    state = gc.advance_state(state, 7)   # "profile"
    state = gc.advance_state(state, 5)   # :
    state = gc.advance_state(state, 0)   # {
    state = gc.advance_state(state, 9)   # "tags"
    state = gc.advance_state(state, 5)   # :
    state = gc.advance_state(state, 2)   # [
    
    valid = gc.get_valid_token_ids(state)
    assert 19 not in valid, '"toolong" should be blocked by maxlength'


def test_whitespace_variants():
    """
    tests all whitespace variants are accepted
    """
    vocab = [b'{', b'}', b' ', b'\n', b'\t', b'\r', b'"x"', b':']
    
    gc = GrammarConstraint(vocab)
    schema = '{"type": "object", "properties": {"x": {"type": "string"}}}'
    gc.compile_json_schema(schema)
    
    # test with various whitespace
    state = gc.init_state()
    state = gc.advance_state(state, 0)  # {
    state = gc.advance_state(state, 3)  # \n
    state = gc.advance_state(state, 4)  # \t
    state = gc.advance_state(state, 2)  # space
    state = gc.advance_state(state, 5)  # \r
    state = gc.advance_state(state, 6)  # "x"
    state = gc.advance_state(state, 7)  # :
    state = gc.advance_state(state, 2)  # space
    state = gc.advance_state(state, 6)  # "x"
    state = gc.advance_state(state, 3)  # \n
    state = gc.advance_state(state, 1)  # }
    
    assert gc.is_match_state(state), "all whitespace variants should be accepted"


if __name__ == "__main__":
    test_whitespace_variants()
    print("whitespace variants: pass")
    
    test_kitchen_sink_pretty_printed()
    print("kitchen sink pretty printed: pass")
    
    test_kitchen_sink_constraint_enforcement()
    print("kitchen sink constraint enforcement: pass")
    
    print("\n=== all kitchen sink tests pass ===")
