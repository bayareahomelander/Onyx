"""
json schema constrained generation example

demonstrates using onyx's grammarconstraint engine to enforce
a complex json schema during token-by-token generation.
"""
import json
from onyx_rust import GrammarConstraint


def main():
    # define a realistic schema with multiple constraint types
    schema = {
        "type": "object",
        "required": ["user_id", "profile"],
        "properties": {
            "user_id": {"type": "integer"},
            "profile": {
                "type": "object",
                "required": ["name", "tags"],
                "properties": {
                    "name": {
                        "type": "string",
                        "pattern": "^[A-Z][a-z]+$"
                    },
                    "age": {
                        "type": ["number", "null"]
                    },
                    "status": {
                        "enum": ["active", "suspended"]
                    },
                    "tags": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {"type": "string", "maxLength": 5}
                    }
                }
            }
        }
    }

    # simulated tokenizer vocabulary
    # in production, this comes from your tokenizer (e.g., tokenizer.get_vocab())
    vocab = [
        b'{', b'}', b'[', b']', b',', b':', b'"',     # structural tokens
        b'\n', b'\t', b' ',                              # whitespace
        b'"user_id"', b'"profile"',                      # object keys
        b'"name"', b'"age"', b'"status"', b'"tags"',     # nested keys
        b'42',                                           # integer value
        b'"Alice"',                                      # valid name (uppercase + lowercase)
        b'"bob"',                                        # invalid name (no uppercase start)
        b'null',                                         # null literal
        b'"active"', b'"suspended"',                     # enum values
        b'"api"', b'"web"', b'"cli"',                    # valid tags (<=5 chars)
        b'"toolong"',                                    # invalid tag (>5 chars)
    ]

    gc = GrammarConstraint(vocab)
    gc.compile_json_schema(json.dumps(schema))

    # --- mock generation loop ---
    # simulates how you would use this in a real generation pipeline.
    # at each step, the engine tells you which tokens are valid.

    print("schema compiled. starting mock generation...\n")

    # target output (pretty-printed):
    # {
    #   "user_id": 42,
    #   "profile": {
    #     "name": "Alice",
    #     "tags": ["api", "web"]
    #   }
    # }

    token_sequence = [
        (0, '{'),
        (7, '\\n'),
        (8, '\\t'),
        (10, '"user_id"'),
        (5, ':'),
        (9, ' '),
        (16, '42'),
        (4, ','),
        (7, '\\n'),
        (8, '\\t'),
        (11, '"profile"'),
        (5, ':'),
        (9, ' '),
        (0, '{'),
        (7, '\\n'),
        (8, '\\t'),
        (8, '\\t'),
        (12, '"name"'),
        (5, ':'),
        (9, ' '),
        (17, '"Alice"'),
        (4, ','),
        (7, '\\n'),
        (8, '\\t'),
        (8, '\\t'),
        (15, '"tags"'),
        (5, ':'),
        (9, ' '),
        (2, '['),
        (22, '"api"'),
        (4, ','),
        (9, ' '),
        (23, '"web"'),
        (3, ']'),
        (7, '\\n'),
        (8, '\\t'),
        (1, '}'),
        (7, '\\n'),
        (1, '}'),
    ]

    state = gc.init_state()

    for token_id, label in token_sequence:
        valid_ids = gc.get_valid_token_ids(state)

        if token_id not in valid_ids:
            print(f"  BLOCKED: token {token_id} ({label}) not in valid set")
            return

        state = gc.advance_state(state, token_id)
        print(f"  accepted: {label}")

    if gc.is_match_state(state):
        print("\ngeneration complete. output is valid json matching schema.")
    else:
        print("\ngeneration incomplete. schema not fully satisfied.")

    # --- demonstrate constraint enforcement ---
    print("\n--- constraint enforcement demo ---\n")

    gc.reset()
    state = gc.init_state()
    state = gc.advance_state(state, 0)   # {
    state = gc.advance_state(state, 11)  # "profile"
    state = gc.advance_state(state, 5)   # :
    state = gc.advance_state(state, 0)   # {
    state = gc.advance_state(state, 12)  # "name"
    state = gc.advance_state(state, 5)   # :

    valid = gc.get_valid_token_ids(state)
    alice_valid = 17 in valid   # "Alice"
    bob_valid = 18 in valid     # "bob"
    print(f'"Alice" valid for name: {alice_valid}')
    print(f'"bob" valid for name:   {bob_valid}')

    # jump to tags, test maxitems
    gc.reset()
    state = gc.init_state()
    state = gc.advance_state(state, 0)   # {
    state = gc.advance_state(state, 11)  # "profile"
    state = gc.advance_state(state, 5)   # :
    state = gc.advance_state(state, 0)   # {
    state = gc.advance_state(state, 15)  # "tags"
    state = gc.advance_state(state, 5)   # :
    state = gc.advance_state(state, 2)   # [

    # check that "toolong" is blocked by maxlength=5
    valid = gc.get_valid_token_ids(state)
    toolong_blocked = 25 not in valid
    print(f'"toolong" blocked in tags: {toolong_blocked}')

    # add 3 tags (maxitems=3)
    state = gc.advance_state(state, 22)  # "api"
    state = gc.advance_state(state, 4)   # ,
    state = gc.advance_state(state, 23)  # "web"
    state = gc.advance_state(state, 4)   # ,
    state = gc.advance_state(state, 24)  # "cli"

    valid = gc.get_valid_token_ids(state)
    comma_blocked = 4 not in valid  # can't add 4th item
    close_allowed = 3 in valid      # ] is valid
    print(f'4th item blocked (maxitems=3): {comma_blocked}')
    print(f'close bracket allowed: {close_allowed}')


if __name__ == "__main__":
    main()
