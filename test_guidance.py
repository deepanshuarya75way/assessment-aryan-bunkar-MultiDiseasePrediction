from guidance.guidance_engine import GuidanceEngine


engine = GuidanceEngine()

guidance = engine.get_guidance(
    disease_id=1,
    probability=0.78
)


print("\nProbability Level:")
print(guidance["probability_level"])


print("\nWhat to consume:")
for item in guidance["consume"]:
    print("-", item)


print("\nWhat to limit:")
for item in guidance["limit"]:
    print("-", item)


print("\nLifestyle:")
for item in guidance["lifestyle"]:
    print("-", item)


print("\nWhen to seek care:")
for item in guidance["seek_care"]:
    print("-", item)

print("\nSources:")
for source in guidance["sources"]:
    print("-", source["organization"], "-", source["title"])