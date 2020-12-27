#
# Function by Antti Haapala via https://stackoverflow.com/a/18178379
#
import ast
import decimal


def turn_into_dict(source: str) -> dict:
    """Transform string into dictionary."""
    tree = ast.parse(source, mode="eval")

    # using the NodeTransformer, you can also modify the nodes in the tree,
    # however in this example NodeVisitor could do as we are raising exceptions
    # only.
    class Transformer(ast.NodeTransformer):
        ALLOWED_NAMES = set(
            ["Decimal", "None", "False", "false", "True", "true", "null"]
        )
        ALLOWED_NODE_TYPES = set(
            [
                "Expression",  # a top node for an expression
                "Tuple",  # makes a tuple
                "Call",  # a function call (hint, Decimal())
                "Name",  # an identifier...
                "Load",  # loads a value of a variable with given identifier
                "Str",  # a string literal
                "Num",  # allow numbers too
                "List",  # and list literals
                "Dict",  # and dicts...
                "UnaryOp",
                "USub",
            ]
        )

        def visit_Name(self, node):
            if node.id not in self.ALLOWED_NAMES:
                raise RuntimeError(f"Name access to {node.id} is not allowed")

            # traverse to child nodes
            return self.generic_visit(node)

        def generic_visit(self, node):
            nodetype = type(node).__name__
            if nodetype not in self.ALLOWED_NODE_TYPES:
                raise RuntimeError(f"Invalid expression: {nodetype} not allowed")

            return ast.NodeTransformer.generic_visit(self, node)

    transformer = Transformer()

    # raises RuntimeError on invalid code
    transformer.visit(tree)

    # compile the ast into a code object
    clause = compile(tree, "<AST>", "eval")

    # make the globals contain only the Decimal class,
    # and eval the compiled object
    result = eval(
        clause,
        {
            "Decimal": decimal.Decimal,
            "false": False,
            "true": True,
            "null": None,
            "UnaryOp": ast.UnaryOp,
        },
    )
    return result
