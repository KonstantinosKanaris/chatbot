import json
import os
from typing import Any, Dict, List

from chatbot import logger


class CornellDialogsParser:
    """Extracts the query-response sequence pairs from the
    cornell-dialogs movie corpus.

    Example:
        >>> from chatbot.data.parsers import CornellDialogsParser
        >>> conversations = CornellDialogsParser.parse_conversations(
        ...     filepath="path-to-utterances.jsonl"
        ... )
        >>> pairs = CornellDialogsParser.extract_sequence_pairs(
        ...     conversations=conversations
        ... )
        >>> pairs
        [['They do to!', 'They do not!'],
         ['She okay?', 'I hope so.'],
         ['Wow', "Let's go."],
         ['No', "Okay -- you're gonna need to learn how to lie."],
         ['What good stuff?', 'The "real you".'],
         ['The "real you".', 'Like my fear of wearing pastels?'],
         ['do you listen to this crap?', 'What crap?'],
         ...
        ]
    """

    @classmethod
    def parse_conversations(cls, filepath: str) -> Dict[str, Any]:
        """Extracts the cornell movie dialog conversations from
        a json file.

        Each conversation consists of multiple lines (sequences).

        Args:
            filepath (str): The path to the target file.

        Returns:
            Dict[str, Any]: The cornell movie dialog conversations.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"The provided file '{filepath}' does not exist."
            )

        lines = {}
        conversations: Dict[str, Any] = {}
        logger.info(f"Extracting conversations from {filepath}")
        with open(file=filepath, mode="r", encoding="iso-8859-1") as f:
            for i, line in enumerate(f):
                line_json = json.loads(s=line)

                # Extract fields for line object
                line_obj = {
                    "line_id": line_json["id"],
                    "character_id": line_json["speaker"],
                    "text": line_json["text"],
                }
                lines[line_obj["line_id"]] = line_obj

                # Extract fields for conversation object
                if line_json["conversation_id"] not in conversations:
                    conv_obj = {
                        "conversation_id": line_json["conversation_id"],
                        "movie_id": line_json["meta"]["movie_id"],
                        "lines": [line_obj],
                    }
                else:
                    conv_obj = conversations[line_json["conversation_id"]]
                    conv_obj["lines"].append(line_obj)
                    conv_obj["lines"] = sorted(
                        conv_obj["lines"], key=lambda item: item["line_id"]
                    )
                conversations[conv_obj["conversation_id"]] = conv_obj

        return conversations

    @classmethod
    def extract_sequence_pairs(
        cls, conversations: Dict[str, Any]
    ) -> List[List[str]]:
        """Extracts pairs of query-response sequences from the
        conversations dictionary.

        Args:
            conversations (Dict[str, Any]): The conversations'
                dictionary.

        Returns:
            List[List[str]]: All the query-answer sequence pairs.
        """
        logger.info("Extracting query-response sequence pairs ...")
        qa_pairs = []
        for conversation in conversations.values():
            for i in range(len(conversation["lines"]) - 1):
                query_sequence = conversation["lines"][i]["text"].strip()
                target_sequence = conversation["lines"][i + 1]["text"].strip()

                # Filter wrong samples (if one of the lists is empty)
                if query_sequence and target_sequence:
                    qa_pairs.append([query_sequence, target_sequence])
        return qa_pairs
