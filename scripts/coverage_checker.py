import argparse
import sys
from enum import Flag, StrEnum, auto
from typing import IO, Iterator, Union


class ItemType(StrEnum):
    CRATE = "crate"
    MOD = "mod"
    STRUCT = "struct"
    ENUM = "enum"
    TYPE = "type"
    TRAIT = "trait"
    FN = "fn"
    CONST_FN = "const fn"
    ASYNC_FN = "async fn"


class ItemTypeFlag(Flag):
    CRATE = auto()
    MOD = auto()
    STRUCT = auto()
    ENUM = auto()
    TYPE = auto()
    TRAIT = auto()
    FN = auto()
    CONST_FN = auto()
    ASYNC_FN = auto()

    ALL_FN = FN | CONST_FN | ASYNC_FN


class Visibility(StrEnum):
    PUB = "pub"
    PUB_CRATE = "pub(crate)"
    PUB_SELF = "pub(self)"
    PUB_SUPER = "pub(super)"


class InputLine:
    def __init__(self, line: str):
        line_parts = line.split("\t")

        self.data = line_parts[0]
        self.comment = line_parts[-1] if len(line_parts) > 1 else None
        self.tab_count = line.count("\t") if self.comment is not None else None

    def has_comment(self) -> bool:
        return self.comment is not None

    def get_full_line(self) -> str:
        return self.data + ("\t" * self.tab_count + self.comment) if self.has_comment() else ""

    def __str__(self) -> str:
        return f"InputLine(data='{self.data}', comment='{self.comment}', tab_count={self.tab_count})"


class Entry:
    def __init__(self, line: str, item_type: ItemType, find_index: int):
        self.line = InputLine(line)
        self.item_type = item_type
        self.name = self._find_name(self.line.data)
        self.nest_level = self._compute_nest_level(find_index)
        self.visibility = self._find_visibility(self.line.data)

    def _compute_nest_level(self, find_index: int) -> int:
        indent_size = 4
        return (find_index + 2) // indent_size  # +2 as we match with "─ " at the start

    def _find_visibility(self, data_line: str) -> Visibility:
        tokens = data_line.split()

        if tokens[0] == "crate":
            # Crate has no visibility specifier, set pub for convinience
            return Visibility.PUB

        tokens.reverse()
        for token in tokens:
            token = token.strip()
            if token.startswith("pub"):
                return Visibility(token)

        raise RuntimeError("Visibility not detected for: " + data_line)

    def _find_name(self, data_line: str) -> str:
        tokens = data_line.split()

        if tokens[0] == "crate":  # Crate name does not end with colon
            return f"{tokens[0]}_{tokens[1]}"  # crate <module_name>

        tokens.reverse()
        for ndx, token in enumerate(tokens):
            token = token.strip()
            if token.endswith(":"):
                return f"{tokens[ndx + 1]}_{token[:-1]}"

    def __str__(self) -> str:
        return f"Entry(line={self.line}, name={self.name}, item_type={self.item_type}, nest_level={self.nest_level}, visibility={self.visibility})"


class Indexer:
    def __init__(self, ndx_count: int):
        self.value = None
        self.ndx_count = ndx_count

    def _raise_if_invalid_range(self, new_val: int) -> None:
        if new_val >= self.ndx_count:
            raise IndexError("Indexer out of range")

    def next(self) -> int:
        if self.value is None:
            self.value = 0
        else:
            new_value = self.value + 1
            self._raise_if_invalid_range(new_value)
            self.value = new_value

        return self.value

    def next_no_inc(self) -> int:
        if self.value is None:
            raise RuntimeError("Indexer not started")

        new_value = self.value + 1
        self._raise_if_invalid_range(new_value)
        return new_value


class TreeNode:
    def __init__(self, entries: list[Entry], ndx: Indexer, parent: Union["TreeNode", None] = None):
        self.metadata = set()
        try:
            entry = entries[ndx.next()]

            self.line = entry.line
            self.name = entry.name
            self.unique_name = f"{parent.unique_name}::{self.name}" if parent is not None else self.name
            self.item_type = entry.item_type
            self.visibility = entry.visibility
            self.parent = parent
            self.children = []

            nest_level = entry.nest_level

            while True:
                next_entry = entries[ndx.next_no_inc()]

                if next_entry.nest_level > nest_level:
                    self.children.append(TreeNode(entries, ndx, self))
                else:
                    return

        except IndexError:
            return

    def traverse_depth_first_pre_order(self) -> Iterator["TreeNode"]:
        yield self

        for child in self.children:
            yield from child.traverse_depth_first_pre_order()

    def traverse_depth_first_post_order(self) -> Iterator["TreeNode"]:
        for child in self.children:
            yield from child.traverse_depth_first_post_order()

        yield self


class Tree:
    def __init__(self, entries: list[Entry]):
        self.root = TreeNode(entries, Indexer(len(entries)))

    def write(self, target: IO[str]) -> None:
        for node in self.root.traverse_depth_first_pre_order():
            target.write(f"{node.line.get_full_line()}\n")

    def filter_visibility(self, visibility: Visibility) -> None:
        for node in self.root.traverse_depth_first_pre_order():
            children_to_remove = []
            for ndx, child in enumerate(node.children):
                if child.visibility != visibility.value:
                    children_to_remove.append(ndx)

            for ndx in reversed(children_to_remove):
                del node.children[ndx]

    def filter_item_type(self, item_type_flag: ItemTypeFlag) -> None:
        for node in self.root.traverse_depth_first_post_order():
            children_to_remove = []
            # Check if any child has the matching item type
            for ndx, child in enumerate(node.children):
                any_flag_present = False

                # ItemTypeFlag can consist of multiple ItemType
                for flag in item_type_flag:
                    item_type = ItemType[flag.name]
                    if child.item_type == item_type:
                        any_flag_present = True
                        break

                if any_flag_present or ("has_matching_type" in child.metadata):
                    node.metadata.add("has_matching_type")
                else:
                    children_to_remove.append(ndx)

            # Remove children that do not match the item type
            for ndx in reversed(children_to_remove):
                del node.children[ndx]

    def _find_node_by_unique_name(self, unique_name: str) -> TreeNode:
        for node in self.root.traverse_depth_first_pre_order():
            if node.unique_name == unique_name:
                return node

        raise RuntimeError(f"Node with unique name '{unique_name}' not found")

    def add_comments_from_reference(self, reference_tree: "Tree") -> None:
        """
        Add comments from the reference tree to the current tree.
        Comments are added to nodes with matching unique names.
        """
        for ref_node in reference_tree.root.traverse_depth_first_pre_order():
            try:
                node = self._find_node_by_unique_name(ref_node.unique_name)
                if ref_node.line.has_comment():
                    node.line.comment = ref_node.line.comment
                    node.line.tab_count = ref_node.line.tab_count
            except RuntimeError:
                sys.stderr.write(
                    f"Warning: Reference node with unique name '{ref_node.unique_name}' not found in the current tree.\n"
                )

    def count_leaf_nodes(self) -> int:
        count = 0
        for node in self.root.traverse_depth_first_pre_order():
            if not node.children:
                count += 1
        return count

    def count_leaf_nodes_with_comment(self, comment: str) -> int:
        count = 0
        for node in self.root.traverse_depth_first_pre_order():
            if not node.children and node.line.has_comment() and node.line.comment.startswith(comment):
                count += 1
        return count


def process_line(line: str) -> Entry:
    if line.startswith("crate"):
        return Entry(line, ItemType.CRATE, 0)

    for item_type in ItemType:
        found = line.find(f"─ {item_type.value} ")
        if found != -1:
            return Entry(line, item_type, found)

    raise RuntimeError("No known item type found for parsed line: " + line)


def process_input(input_lines: list[str]) -> list[Entry]:
    data = []

    for line in input_lines:
        # We cannot remove leading spaces, as they are part of the structure
        line = line.rstrip()
        if len(line) == 0:
            continue

        entry = process_line(line)
        data.append(entry)

    return data


def process_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter and compare Rust crate structure")
    parser.add_argument(
        "-v",
        "--visibility",
        type=str,
        choices=[v.name for v in Visibility],
        default=Visibility.PUB.name,
        help="Visibility to filter",
    )
    parser.add_argument(
        "-t",
        "--item-type",
        type=str,
        choices=[v.name for v in ItemTypeFlag],
        default=ItemTypeFlag.ALL_FN.name,
        help="Item type to focus on",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="File to read input from (default: stdin)",
    )
    parser.add_argument(
        "-r",
        "--reference",
        type=str,
        default=None,
        help="Reference result to copy comments from",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="File to write output to (default: stdout)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = process_args()

    if args.input:
        with open(args.input, "r") as f:
            input_lines = f.readlines()
    else:
        input_lines = sys.stdin.readlines()

    tree = Tree(process_input(input_lines))
    tree.filter_visibility(Visibility[args.visibility])
    tree.filter_item_type(ItemTypeFlag[args.item_type])

    if args.reference:
        with open(args.reference, "r") as f:
            reference_lines = f.readlines()

        ref_tree = Tree(process_input(reference_lines))
        tree.add_comments_from_reference(ref_tree)

    if args.output:
        with open(args.output, "w") as f:
            tree.write(f)
    else:
        tree.write(sys.stdout)

    covered = tree.count_leaf_nodes_with_comment("YES")
    all = tree.count_leaf_nodes()
    print(f"Coverage rate: {covered}/{all} ({covered / all * 100:.2f}%)")
