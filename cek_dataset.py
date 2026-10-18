
from pathlib import Path
import pandas as pd

FILES = [
    Path("datasets/indonesian_news_clean.csv"),
    Path("datasets/final_merge_dataset.csv"),
]

COLUMNS = ["Judul", "Content", "Link", "source"]


def analyze_dataset(path):
    print("\n" + "=" * 65)
    print(f"DATASET: {path.name}")
    print("=" * 65)

    if not path.exists():
        print("File tidak ditemukan!")
        return

    stats = {
        "rows": 0,
        "empty_title": 0,
        "empty_content": 0,
        "empty_link": 0,
        "short_content": 0,
        "content_chars": 0,
        "max_content_length": 0,
        "duplicate_links": 0,
        "nonempty_content": 0,
    }

    seen_links = set()
    sources = set()
    samples = []

    try:
        for chunk in pd.read_csv(
            path,
            usecols=COLUMNS,
            chunksize=20000,
            low_memory=False,
        ):
            for col in COLUMNS:
                chunk[col] = (
                    chunk[col]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )

            titles = chunk["Judul"]
            contents = chunk["Content"]
            links = chunk["Link"]
            content_lengths = contents.str.len()

            stats["rows"] += len(chunk)
            stats["empty_title"] += int((titles == "").sum())
            stats["empty_content"] += int((contents == "").sum())
            stats["empty_link"] += int((links == "").sum())
            stats["short_content"] += int(
                ((contents != "") & (content_lengths < 200)).sum()
            )

            nonempty = contents != ""
            stats["nonempty_content"] += int(nonempty.sum())
            stats["content_chars"] += int(content_lengths[nonempty].sum())

            if not content_lengths.empty:
                stats["max_content_length"] = max(
                    stats["max_content_length"],
                    int(content_lengths.max()),
                )

            for link in links:
                if not link:
                    continue
                if link in seen_links:
                    stats["duplicate_links"] += 1
                else:
                    seen_links.add(link)

            sources.update(source for source in chunk["source"] if source)

            if len(samples) < 3:
                for _, row in chunk.iterrows():
                    if len(samples) == 3:
                        break
                    samples.append(
                        (row["Judul"], len(row["Content"]))
                    )

    except Exception as error:
        print(f"Gagal membaca dataset: {error}")
        return

    nonempty_count = stats["nonempty_content"]
    avg_length = (
        stats["content_chars"] / nonempty_count
        if nonempty_count else 0
    )

    print(f"Jumlah berita          : {stats['rows']:,}")
    print(f"Judul kosong           : {stats['empty_title']:,}")
    print(f"Konten kosong          : {stats['empty_content']:,}")
    print(f"Link kosong            : {stats['empty_link']:,}")
    print(f"Konten kurang 200 char : {stats['short_content']:,}")
    print(f"Duplikasi link         : {stats['duplicate_links']:,}")
    print(f"Jumlah sumber berita   : {len(sources):,}")
    print(f"Rata-rata panjang konten: {avg_length:,.0f} karakter")
    print(f"Konten terpanjang      : {stats['max_content_length']:,} karakter")

    print("\nContoh tiga berita:")
    for title, content_length in samples:
        print(f"- {title[:100]} ({content_length:,} karakter)")


if __name__ == "__main__":
    for file in FILES:
        analyze_dataset(file)
