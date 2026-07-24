#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace md {

struct HighscoreEntry {
    std::array<char, 3> initials{{'A', 'A', 'A'}};
    int score = 0;
};

/// A small persistent top-scores table (arcade style). Stored as plain text in
/// the platform's app-data location; degrades to empty if it cannot be read.
class HighscoreTable {
  public:
    static constexpr std::size_t capacity = 10;

    void load();       // (re)read the table from disk
    void save() const; // write the table to disk

    /// True if `score` would earn a place on the table.
    [[nodiscard]] bool qualifies(int score) const;

    /// Insert an entry, keep the top `capacity` sorted by score, persist, and
    /// return the rank (0-based) the new entry landed at.
    std::size_t insert(std::array<char, 3> initials, int score);

    [[nodiscard]] const std::vector<HighscoreEntry>& entries() const noexcept { return entries_; }

  private:
    void fill_defaults(); // seed a full board when there is no saved table yet

    std::vector<HighscoreEntry> entries_;
};

} // namespace md
