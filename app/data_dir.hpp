// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Jens Köhler
// Assisted-by: Claude Code (Anthropic)
#pragma once

/// Where this application keeps its per-user data — and the one rule the game
/// and the trainer must agree on, because they write and read the same files.
///
/// They did not agree. `main` sets an organisation name *and* an application
/// name, both "MissileDefense", and Qt builds `AppDataLocation` and
/// `AppLocalDataLocation` out of **both**. So the game resolved
/// `~/.local/share/MissileDefense/MissileDefense` while
/// `missile_defense.runs.paths.data_home` appended the name once and wrote to
/// `~/.local/share/MissileDefense`. Two promoted models sat in the trainer's
/// `models/` and WATCH AI showed none of them: the game was reading a directory
/// that did not exist.
///
/// `GenericDataLocation` plus the application name is the fix, because it is the
/// same rule the Python side follows on all three platforms —
/// `~/.local/share`, `%LOCALAPPDATA%` and `~/Library/Application Support`.
///
/// The organisation name stays where it is. It is what routes `QSettings` to
/// `~/.config/MissileDefense/MissileDefense.conf`, and a Qt application without
/// one files its settings under "Unknown Organization" and an access error. It
/// simply no longer decides where *data* lives.

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QStandardPaths>
#include <QString>
#include <QStringList>

namespace md {

/// Mirrors `QGuiApplication::setApplicationName` in `app/main.cpp` and
/// `APP_NAME` in `missile_defense/runs/paths.py`. If one moves, all three move.
inline constexpr auto app_data_name = "MissileDefense";

/// The directory the game shares with the trainer. Created on demand, because
/// every caller is about to read or write a file in it.
inline QString app_data_dir() {
    const QString dir =
        QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation) + "/" + app_data_name;
    QDir().mkpath(dir);
    return dir;
}

/// Move anything the old doubled path collected into the directory that
/// replaced it, once, at startup.
///
/// File by file rather than directory by directory, and never over something
/// already there: the new location is the trainer's, it may already hold a
/// `models/` and a `runs/` worth of work, and a migration that clobbered those
/// would be a far worse bug than the one it is repairing. Anything that cannot
/// be moved is left alone and skipped — a stale copy of a high-score table is
/// not worth refusing to start over.
inline void migrate_legacy_data_dir() {
    const QString destination = app_data_dir();
    const QString canonical = QDir{destination}.canonicalPath();
    // *Both* of Qt's application locations, because the two were not even
    // consistent with each other: highscores.cpp used `AppDataLocation` and
    // game_window.cpp used `AppLocalDataLocation`. On Linux and macOS those are
    // the same directory and the second pass finds nothing left to do; on
    // Windows they are genuinely different — `AppData/Roaming/...` held the
    // high scores and `AppData/Local/...` the trainer record — so both have to
    // be swept or Windows players lose their table.
    for (const QStandardPaths::StandardLocation location :
         {QStandardPaths::AppDataLocation, QStandardPaths::AppLocalDataLocation}) {
        const QString legacy = QStandardPaths::writableLocation(location);
        QDir source{legacy};
        if (legacy.isEmpty() || !source.exists()) {
            continue;
        }
        // Compared as canonical paths and not as strings: on Windows these
        // differ in separator and case from the one built above, and on macOS a
        // symlinked home would make two spellings of one directory. Moving a
        // file onto itself is how a migration deletes a high-score table.
        if (source.canonicalPath() == canonical) {
            continue;
        }
        for (const QString& name : source.entryList(QDir::Files | QDir::Hidden)) {
            const QString target = destination + "/" + name;
            if (!QFile::exists(target)) {
                QFile::rename(source.filePath(name), target);
            }
        }
        // Only if the move emptied it, and only the leaf: a directory still
        // holding something is one this function did not understand, and
        // removing it would be guessing.
        source.cdUp();
        source.rmdir(QFileInfo{legacy}.fileName());
    }
}

} // namespace md
