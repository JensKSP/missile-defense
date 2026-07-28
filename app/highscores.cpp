// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#include "highscores.hpp"

#include <QDir>
#include <QStandardPaths>
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <string>

namespace md {

namespace {

std::string file_path() {
    const QString dir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir().mkpath(dir);
    return (dir + "/highscores.txt").toStdString();
}

void sort_desc(std::vector<HighscoreEntry>& v) {
    std::ranges::stable_sort(
        v, [](const HighscoreEntry& a, const HighscoreEntry& b) { return a.score > b.score; });
    if (v.size() > HighscoreTable::capacity) {
        v.resize(HighscoreTable::capacity);
    }
}

} // namespace

void HighscoreTable::fill_defaults() {
    entries_ = {
        HighscoreEntry{.initials = {'A', 'C', 'E'}, .score = 10000},
        HighscoreEntry{.initials = {'B', 'E', 'N'}, .score = 9000},
        HighscoreEntry{.initials = {'C', 'A', 'T'}, .score = 8000},
        HighscoreEntry{.initials = {'D', 'A', 'X'}, .score = 7000},
        HighscoreEntry{.initials = {'E', 'V', 'E'}, .score = 6000},
        HighscoreEntry{.initials = {'F', 'O', 'X'}, .score = 5000},
        HighscoreEntry{.initials = {'G', 'U', 'S'}, .score = 4000},
        HighscoreEntry{.initials = {'H', 'A', 'L'}, .score = 3000},
        HighscoreEntry{.initials = {'I', 'V', 'Y'}, .score = 2000},
        HighscoreEntry{.initials = {'J', 'A', 'X'}, .score = 1000},
    };
}

void HighscoreTable::load() {
    entries_.clear();
    std::ifstream in(file_path());
    if (in) {
        std::string initials;
        int score = 0;
        // `setw` bounds the token, and the loop bounds the table. Neither guards
        // against a player: this file is three characters and a number per line,
        // and every board is sorted down to `capacity` two lines below. They
        // guard against a file that is no longer that — a truncated sync, a
        // stray binary, an editor that saved something else here — which would
        // otherwise be read into an unbounded string and an unbounded vector
        // before anything looked at it. `capacity` rows are all that survive, so
        // reading more than a few is already work with no purpose.
        constexpr std::size_t most = HighscoreTable::capacity * 8;
        while (entries_.size() < most &&
               in >> std::setw(static_cast<int>(HighscoreEntry{}.initials.size()) + 1) >>
                   initials >> score) {
            HighscoreEntry entry;
            // Copy as many initials as the file supplied, space-padding the rest.
            entry.initials.fill(' ');
            std::copy_n(initials.begin(), std::min(initials.size(), entry.initials.size()),
                        entry.initials.begin());
            entry.score = score;
            entries_.push_back(entry);
        }
        sort_desc(entries_);
    }
    if (entries_.empty()) { // fresh install: seed a full default board
        fill_defaults();
        save();
    }
}

void HighscoreTable::save() const {
    std::ofstream out(file_path(), std::ios::trunc);
    if (!out) {
        return;
    }
    for (const auto& entry : entries_) {
        out << std::string(entry.initials.begin(), entry.initials.end()) << ' ' << entry.score
            << '\n';
    }
}

bool HighscoreTable::qualifies(int score) const {
    if (score <= 0) {
        return false;
    }
    return entries_.size() < capacity || score > entries_.back().score;
}

std::size_t HighscoreTable::insert(std::array<char, 3> initials, int score) {
    entries_.push_back(HighscoreEntry{.initials = initials, .score = score});
    sort_desc(entries_);
    save();
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        if (entries_[i].score == score && entries_[i].initials == initials) {
            return i; // rank of the freshly-inserted entry
        }
    }
    return 0;
}

} // namespace md
