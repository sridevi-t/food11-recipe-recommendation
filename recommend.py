"""
recommend.py
------------
Food-specific recipe recommendation system.
"""

import argparse

from utils import load_recipes_db


def get_available_cuisines(db=None):
    """Return all unique cuisine types in the recipe database."""
    if db is None:
        db = load_recipes_db()

    cuisines = set()

    for recipes in db.values():
        for recipe in recipes:
            if recipe.get("cuisine"):
                cuisines.add(recipe["cuisine"])

    return sorted(cuisines)


def get_recommendations(
    food_class,
    vegetarian=None,
    cuisine=None,
    top_n=5,
    db=None
):
    """
    Recommend recipes for the predicted Food-11 class.

    The recipe database uses the same names as the model classes:
        pizza -> pizza
        sushi -> sushi
        ice_cream -> ice_cream
        chicken_curry -> chicken_curry
    """

    if db is None:
        db = load_recipes_db()

    # Clean the predicted class name
    food_class = food_class.strip()

    # Find matching food category
    recipe_category = None

    # Exact match
    if food_class in db:
        recipe_category = food_class

    # Case-insensitive match
    if recipe_category is None:
        for key in db:
            if key.strip().lower() == food_class.lower():
                recipe_category = key
                break

    # No matching food
    if recipe_category is None:
        return []

    candidates = db[recipe_category]

    # Vegetarian / Non-Vegetarian filter
    if vegetarian is not None:
        candidates = [
            recipe
            for recipe in candidates
            if recipe.get("vegetarian") == vegetarian
        ]

    # Cuisine filter
    if cuisine:
        cuisine_lower = cuisine.strip().lower()

        candidates = [
            recipe
            for recipe in candidates
            if recipe.get("cuisine", "").strip().lower()
            == cuisine_lower
        ]

    return candidates[:top_n]


def _parse_veg_flag(value):
    if value is None:
        return None

    value = value.strip().lower()

    if value in ("veg", "vegetarian", "true", "yes"):
        return True

    if value in (
        "non-veg",
        "nonveg",
        "non vegetarian",
        "non-vegetarian",
        "false",
        "no"
    ):
        return False

    raise argparse.ArgumentTypeError(
        f"Unrecognized vegetarian flag: {value}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Recommend recipes for a predicted food class."
    )

    parser.add_argument(
        "--food_class",
        required=True,
        help="Food class such as pizza, sushi, ice_cream"
    )

    parser.add_argument(
        "--veg",
        default=None,
        help="veg or non-veg"
    )

    parser.add_argument(
        "--cuisine",
        default=None,
        help="Cuisine such as Indian, Italian, Chinese"
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=5
    )

    args = parser.parse_args()

    veg_flag = _parse_veg_flag(args.veg)

    results = get_recommendations(
        args.food_class,
        vegetarian=veg_flag,
        cuisine=args.cuisine,
        top_n=args.top_n
    )

    if not results:
        print("No recipes matched your filters.")
        return

    print(
        f"\nRecommended recipes for '{args.food_class}':\n"
    )

    for recipe in results:
        tag = (
            "Vegetarian"
            if recipe["vegetarian"]
            else "Non-Vegetarian"
        )

        print(
            f"- {recipe['name']} "
            f"[{recipe['cuisine']} | "
            f"{tag} | "
            f"{recipe['prep_time_mins']} min]"
        )

        print(
            f"  Ingredients: "
            f"{', '.join(recipe['ingredients'])}"
        )

        print(
            f"  How to make: "
            f"{recipe['instructions']}\n"
        )


if __name__ == "__main__":
    main()