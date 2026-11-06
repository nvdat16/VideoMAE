import csv
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from prepare_data import prepare_dataset, stratified_split_files


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'Dataset'
        self.data.mkdir()
        self.work = self.root / 'work'
        (self.data / 'Diving48_vocab.json').write_text(json.dumps([['Zulu'], ['Alpha'], ['Empty']]))
        entries = [{'vid_name': f'video_{i:02d}', 'label': i % 2} for i in range(20)]
        (self.data / 'Diving48_V2_train.json').write_text(json.dumps(entries[:14]))
        (self.data / 'Diving48_V2_test.json').write_text(json.dumps(entries[14:]))
        with tarfile.open(self.data / 'Diving48_rgb.tar.gz', 'w:gz') as tar:
            for e in entries:
                info = tarfile.TarInfo(f'rgb/{e["vid_name"]}.mp4')
                info.size = 4
                tar.addfile(info, io.BytesIO(b'fake'))

    def rows(self, prepared):
        with (prepared / 'annotations.csv').open() as f:
            return list(csv.DictReader(f))

    def test_archive_split_and_reuse(self):
        p = prepare_dataset(self.data, self.work)
        rows = self.rows(p)
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({r['video_path'] for r in rows}), 20)
        for r in rows:
            link = p / r['video_path']
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.read_bytes(), b'fake')
        mapping = json.loads((p / 'label_map.json').read_text())['label2id']
        self.assertEqual(mapping, {'Alpha': 0, 'Empty': 1, 'Zulu': 2})
        for split in ('train', 'val', 'test'):
            self.assertTrue((p / split / 'Empty').is_dir())
        before = (p / 'READY.json').stat().st_mtime_ns
        self.assertEqual(prepare_dataset(self.data, self.work), p)
        self.assertEqual(before, (p / 'READY.json').stat().st_mtime_ns)
        with self.assertRaisesRegex(RuntimeError, 'differs'):
            prepare_dataset(self.data, self.work, seed=43)

    def test_missing_video_fails(self):
        prepare_dataset(self.data, self.work)
        next((self.work / 'videos').rglob('*.mp4')).unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'annotated videos missing'):
            prepare_dataset(self.data, self.work)

    def test_extracted_directory_without_archive(self):
        p = prepare_dataset(self.data, self.work)
        (self.data / 'Diving48_rgb.tar.gz').unlink()
        other = prepare_dataset(self.data, self.root / 'other', videos_dir=self.work / 'videos')
        self.assertEqual(self.rows(p), self.rows(other))

    def test_archive_traversal_rejected(self):
        with tarfile.open(self.data / 'Diving48_rgb.tar.gz', 'w:gz') as tar:
            info = tarfile.TarInfo('../outside.mp4')
            info.size = 4
            tar.addfile(info, io.BytesIO(b'evil'))
        with self.assertRaisesRegex(ValueError, 'Unsafe archive'):
            prepare_dataset(self.data, self.work)
        self.assertFalse((self.work / 'outside.mp4').exists())
        self.assertFalse((self.work / 'videos').exists())

    def test_rare_classes_and_determinism(self):
        self.assertEqual(stratified_split_files([]), ([], [], []))
        self.assertEqual(tuple(map(len, stratified_split_files(['a']))), (1, 0, 0))
        self.assertEqual(tuple(map(len, stratified_split_files(['a', 'b']))), (1, 0, 1))
        self.assertEqual(tuple(map(len, stratified_split_files(['a', 'b', 'c']))), (1, 1, 1))
        x = [str(i) for i in range(100)]
        a = stratified_split_files(x)
        self.assertEqual(tuple(map(len, a)), (70, 15, 15))
        self.assertEqual(a, stratified_split_files(x[::-1]))
        self.assertFalse(set(a[0]) & set(a[1]))
        self.assertFalse(set(a[0]) & set(a[2]))
        self.assertFalse(set(a[1]) & set(a[2]))


if __name__ == '__main__':
    unittest.main()
