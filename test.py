import sqlite3

print("Connecting to Lightroom catalogue database...")
conn = sqlite3.connect("lightroom catalogue/lightroom catalogue.lrcat")
cursor = conn.cursor()
print("Connected successfully!\n")

# # List all tables
# cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# tables = cursor.fetchall()
# for table in tables:
#     print(table[0])

print("Querying Adobe_imageDevelopSettings table...")

# Option 1: Use OFFSET to skip rows (e.g., OFFSET 5 gets the 6th image)
# Change OFFSET value to get different images: OFFSET 0 = first, OFFSET 1 = second, etc.
OFFSET_VALUE = 0  # Change this to get a different image

# Option 2: Filter by specific image ID (set to None to use OFFSET instead)
# Set to a specific image ID number to query that image directly
IMAGE_ID = None  # e.g., IMAGE_ID = 12345

if IMAGE_ID is not None:
    # Query specific image by ID
    cursor.execute("""
        SELECT image, text
        FROM Adobe_imageDevelopSettings
        WHERE image = ?
    """, (IMAGE_ID,))
    print(f"Querying for specific image ID: {IMAGE_ID}")
else:
    # Query with OFFSET to get a specific position in the results
    cursor.execute("""
        SELECT image, text
        FROM Adobe_imageDevelopSettings
        LIMIT 1 OFFSET ?
    """, (OFFSET_VALUE,))
    print(f"Querying with OFFSET {OFFSET_VALUE} (getting image at position {OFFSET_VALUE + 1})")

rows = cursor.fetchall()
print(f"Found {len(rows)} rows\n")
print("=" * 80)

for i, row in enumerate(rows, 1):
    image_id = row[0]
    settings = row[1]  # This is XML/XMP data
    print(f"\nRow {i}:")
    print(f"  Image ID: {image_id}")
    print(f"  Settings length: {len(settings)} characters")
    print(f"  Settings preview (first 200 chars): {settings[:200]}...")
    if len(settings) > 200:
        print(f"  Full settings:")
        print(f"  {settings}")
    print("-" * 80)

print(f"\nTotal rows processed: {len(rows)}")
conn.close()
print("Database connection closed.")