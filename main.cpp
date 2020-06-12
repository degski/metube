
// MIT License
//
// Copyright (c) 2020 degski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#if defined( _MSC_VER )

#    ifndef NOMINMAX
#        define NOMINMAX
#    endif

#    ifndef _AMD64_
#        define _AMD64_
#    endif

#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN_DEFINED
#        define WIN32_LEAN_AND_MEAN
#    endif

#    include <fcntl.h>
#    include <io.h>

#    include <windef.h>
#    include <WinBase.h>

#    include <xmmintrin.h>
#    include <emmintrin.h>
#    include <immintrin.h>

#    ifdef WIN32_LEAN_AND_MEAN_DEFINED
#        undef WIN32_LEAN_AND_MEAN_DEFINED
#        undef WIN32_LEAN_AND_MEAN
#    endif

#    define _SILENCE_CXX17_OLD_ALLOCATOR_MEMBERS_DEPRECATION_WARNING
#    define _ENABLE_EXTENDED_ALIGNED_STORAGE

#else

#    include <fcntl.h> // need nix equiv.
#    include <io.h>    // need nix equiv.

#    include <xmmintrin.h>
#    include <emmintrin.h>
#    include <immintrin.h>

#endif

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

#include <array>
#include <atomic>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

#include <initializer_list>
#include <sax/iostream.hpp>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <sstream>
#include <thread>
#include <jthread>
#include <type_traits>
#include <utility>
#include <vector>

/*
    -fsanitize = address

    C:\Program Files\LLVM\lib\clang\10.0.0\lib\windows\clang_rt.asan_cxx-x86_64.lib
    C:\Program Files\LLVM\lib\clang\10.0.0\lib\windows\clang_rt.asan-preinit-x86_64.lib
    C:\Program Files\LLVM\lib\clang\10.0.0\lib\windows\clang_rt.asan-x86_64.lib

    C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\tbb\lib\intel64_win\vc_mt\tbb.lib
*/

#include <sax/prng_sfc.hpp>
#include <sax/uniform_int_distribution.hpp>

#if defined( NDEBUG )
#    define RANDOM 1
#else
#    define RANDOM 0
#endif

namespace ThreadID {
// Creates a new ID.
[[nodiscard]] inline int get ( bool ) noexcept {
    static std::atomic<int> global_id = 0;
    return global_id++;
}
// Returns ID of this thread.
[[nodiscard]] inline int get ( ) noexcept {
    static thread_local int thread_local_id = get ( false );
    return thread_local_id;
}
} // namespace ThreadID

namespace Rng {
// Chris Doty-Humphrey's Small Fast Chaotic Prng.
[[nodiscard]] inline sax::Rng & generator ( ) noexcept {
    if constexpr ( RANDOM ) {
        static thread_local sax::Rng generator ( sax::os_seed ( ), sax::os_seed ( ), sax::os_seed ( ), sax::os_seed ( ) );
        return generator;
    }
    else {
        static thread_local sax::Rng generator ( sax::fixed_seed ( ) + ThreadID::get ( ) );
        return generator;
    }
}
} // namespace Rng

#undef RANDOM

sax::Rng & rng = Rng::generator ( );

#include <sax/utf8conv.hpp>

#if 0
#    define SYSTEM std::wprintf
#else
#    define SYSTEM _wsystem
#endif

#include <hedley.h>

[[nodiscard]] HEDLEY_ALWAYS_INLINE bool equal_m64 ( void const * const a_, void const * const b_ ) noexcept {
    __int64 a;
    memcpy ( &a, a_, sizeof ( __int64 ) );
    __int64 b;
    memcpy ( &b, b_, sizeof ( __int64 ) );
    return a == b;
}

[[nodiscard]] HEDLEY_ALWAYS_INLINE bool is_equal_m128 ( void const * const a_, void const * const b_ ) noexcept {
    return not _mm_movemask_pd ( _mm_cmpneq_pd ( _mm_load_pd ( ( double const * ) a_ ), _mm_load_pd ( ( double const * ) b_ ) ) );
}

union _m128 {

    __m128 m128_m128;
    __m64 m128_m64[ 2 ];
    __int32 m128_m32[ 4 ];
#if defined( _MSC_VER )
    LONG64 m128_long64[ 2 ];
#endif

    _m128 ( ) = default;

    template<typename ValueType, typename = std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m128 )>>
    _m128 ( ValueType const & v_ ) noexcept {
        memcpy ( this, &v_, sizeof ( _m128 ) );
    }
    template<typename ValueType, typename = std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m128 )>>
    _m128 ( ValueType && v_ ) noexcept {
        memcpy ( this, &v_, sizeof ( _m128 ) );
    }

    template<typename HalfSizeValueType, typename = std::enable_if_t<sizeof ( HalfSizeValueType ) >= sizeof ( __m64 )>>
    _m128 ( HalfSizeValueType const & o0_, HalfSizeValueType const & o1_ ) noexcept {
        memcpy ( m128_m64 + 0, &o0_, sizeof ( __m64 ) );
        memcpy ( m128_m64 + 1, &o1_, sizeof ( __m64 ) );
    };

    ~_m128 ( ) = default;

    template<typename ValueType, typename = std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m128 )>>
    std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m128 ), _m128> operator= ( ValueType const & v_ ) noexcept {
        memcpy ( this, &v_, sizeof ( _m128 ) );
        return *this;
    }
    template<typename ValueType, typename = std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m128 )>>
    std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m128 ), _m128> operator= ( ValueType && v_ ) noexcept {
        memcpy ( this, &v_, sizeof ( _m128 ) );
        return *this;
    }

    template<typename ValueType, typename = std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m128 )>>
    [[nodiscard]] bool operator== ( ValueType const & r_ ) const noexcept {
        return is_equal_m128 ( this, &r_ );
    }
    template<typename ValueType, typename = std::enable_if_t<sizeof ( ValueType ) >= sizeof ( __m128 )>>
    [[nodiscard]] bool operator!= ( ValueType const & r_ ) const noexcept {
        return unequal_m128 ( this, &r_ );
    }
};

struct ytid { // I7e1su04gfs
    ytid get ( ) const noexcept { return *this; }

    [[nodiscard]] static constexpr std::size_t size ( ) noexcept { return 16; }

    [[nodiscard]] char * data ( ) noexcept { return reinterpret_cast<char *> ( this ); }
    [[nodiscard]] char const * data ( ) const noexcept { return reinterpret_cast<char const *> ( this ); }

    private:
    _m128 id;
};

#include <leveldb/db.h>
#include <snappystream.hpp>

// Function declarations.

void print_exe_info ( wchar_t * name_, std::size_t size_ ) noexcept;

[[nodiscard]] std::vector<std::string> get_urls ( fs::path const & file_ );

template<typename Stream>
[[nodiscard]] std::string get_line ( Stream & in_ );

[[nodiscard]] std::string get_source ( std::string const & url_ );

[[nodiscard]] std::tuple<fs::path, fs::path> get_source_target ( std::string const & url_ ) noexcept;

[[nodiscard]] std::tuple<std::string, std::string> get_artist_title ( fs::path const & path_ ) noexcept;

int wmain_ ( int argc_, wchar_t * argv_[], wchar_t *[] ) {
    std::vector<std::string> urls = get_urls ( fs::current_path ( ) / "download.txt" );
    print_exe_info ( argv_[ 0 ], urls.size ( ) );
    for ( std::string const & url : urls ) {
        auto [ source, target ] = get_source_target ( url );
        auto [ artist, title ]  = get_artist_title ( source );
        if ( fs::exists ( source ) )
            fs::remove ( source );

        // clang-format on

        SYSTEM ( sax::utf8_to_utf16 (
                     ( std::string{ "youtube-dl.exe --socket-timeout 10 --limit-rate 100K --buffer-size 16K --geo-bypass "
                                    "--extract-audio --audio-format best --audio-quality 0 --yes-playlist -o \"" } +
                       source.filename ( ).string ( ) + std::string{ "\" \"" } + url + std::string{ "\"" } ) )
                     .c_str ( ) );

        std::cout << nl;

        if ( not fs::exists ( target ) ) {

            SYSTEM (
                sax::utf8_to_utf16 (
                    ( std::string{ "opusdec.exe --rate 44100 \"" } + source.filename ( ).string ( ) + std::string{ "\" - | " } +
                      std::string{ "fdkaac.exe --bitrate-mode 5 --gapless-mode 1 --no-timestamp --artist \"" + artist +
                                   "\" --title \"" + title +
                                   "\" --comment \"down-sampled (sox-14.4.2) and re-encoded to m4a from lossy opus-source "
                                   "(fdkaac-1.0.0)\" - -o \"" +
                                   target.filename ( ).string ( ) + "\"" } ) )
                    .c_str ( ) );

            std::cout << nl;

            // clang-format on
        }
    }

    return EXIT_SUCCESS;
}

// Function definitions.

void print_exe_info ( wchar_t * name_, std::size_t size_ ) noexcept {
    _setmode ( _fileno ( stdout ), _O_U16TEXT );
    std::wstring exename = { name_ };
    if ( exename.size ( ) and L':' == exename[ 1 ] and
         std::string::npos != exename.find ( L"\\" ) ) // Is it actually a path in string-form? (doing some light checks)
        exename = fs::path{ exename }.filename ( );
    std::wcout << exename << L"-0.1 - copyright ©" << L" 2020 degski\ndownloading " << size_ << L" files started . . ." << nl;
}

[[nodiscard]] std::vector<std::string> get_urls ( fs::path const & file_ ) {
    std::vector<std::string> list;
    list.reserve ( 32 );
    list.resize ( 1 );
    std::ifstream in ( file_ );
    if ( not in )
        throw ( "cannot open the file : " + file_.string ( ) );
    while ( std::getline ( in, list.back ( ) ) ) {
        if ( list.back ( ).size ( ) )
            list.resize ( list.size ( ) + 1 );
    }
    in.close ( );
    list.resize ( list.size ( ) - list.back ( ).empty ( ) );
    return list;
}

template<typename Stream>
[[nodiscard]] std::string get_line ( Stream & in_ ) {
    std::string l;
    std::getline ( in_, l );
    if ( l.size ( ) )
        l.pop_back ( );
    return l;
}

[[nodiscard]] std::string get_source ( std::string const & url_ ) {
    _wsystem (
        sax::utf8_to_utf16 ( ( std::string ( "youtube-dl.exe --socket-timeout 5 --retries 20 --skip-download --get-title " ) +
                               url_ + std::string ( " > " ) + "c:\\tmp\\tmp.txt" ) )
            .c_str ( ) );
    std::ifstream stream{ fs::path{ "c:\\tmp\\tmp.txt" } };
    std::string filename = { get_line ( stream ) + ".opus" };
    stream.close ( );
    fs::remove ( "c:\\tmp\\tmp.txt" );
    return filename;
}

[[nodiscard]] std::tuple<fs::path, fs::path> get_source_target ( std::string const & url_ ) noexcept {
    fs::path opus_source       = fs::current_path ( ) / get_source ( url_ ),
             m4a_player_target = fs::path{ "F:/Music" } / opus_source.replace_extension ( ".m4a" ).filename ( );
    return { std::move ( opus_source ), fs::exists ( m4a_player_target.parent_path ( ) )
                                            ? std::move ( m4a_player_target )
                                            : opus_source.replace_extension ( ".m4a" ) };
}

[[nodiscard]] std::tuple<std::string, std::string> get_artist_title ( fs::path const & path_ ) noexcept {
    std::size_t pos = path_.stem ( ).string ( ).find ( "-" ), len = std::string::npos != pos;
    if ( std::string::npos != pos ) {
        if ( pos > 0 and ' ' == path_.string ( )[ pos - 1 ] )
            pos -= 1, len += 1;
        if ( pos < ( path_.stem ( ).string ( ).length ( ) - 1 ) - 1 and ' ' == path_.string ( )[ pos + 1 ] )
            len += 1;
    }
    std::string artist = std::string::npos != pos ? path_.stem ( ).string ( ).substr ( 0, pos ) : std::string{ },
                title  = std::string::npos != pos ? path_.stem ( ).string ( ).substr (
                                                       std::string::npos != pos ? pos + len : std::string::npos, std::string::npos )
                                                 : path_.stem ( ).string ( );
    return { std::move ( artist ), std::move ( title ) };
}

#include <sodium.h>

constexpr char PASSWORD[]       = { "Correct Horse Battery Staple" };
constexpr char WRONG_PASSWORD[] = { "Correct Donkey Battery Staple" };

int wmain ( ) {

    sax::enable_virtual_terminal_sequences ( );

    if ( sodium_init ( ) == -1 )
        return 1;

    unsigned char salt[ crypto_pwhash_scryptsalsa208sha256_SALTBYTES ];
    unsigned char key[ crypto_box_SEEDBYTES ];

    randombytes_buf ( salt, sizeof ( salt ) );

    if ( crypto_pwhash_scryptsalsa208sha256 ( key, sizeof key, PASSWORD, strlen ( PASSWORD ), salt,
                                              crypto_pwhash_scryptsalsa208sha256_OPSLIMIT_INTERACTIVE,
                                              crypto_pwhash_scryptsalsa208sha256_MEMLIMIT_INTERACTIVE ) != 0 )
        std::wcout << sax::bg::wred << L"out-of-memory" << nl;

    char hashed_password[ crypto_pwhash_scryptsalsa208sha256_STRBYTES ];

    if ( crypto_pwhash_scryptsalsa208sha256_str ( hashed_password, PASSWORD, strlen ( PASSWORD ),
                                                  crypto_pwhash_scryptsalsa208sha256_OPSLIMIT_INTERACTIVE,
                                                  crypto_pwhash_scryptsalsa208sha256_MEMLIMIT_INTERACTIVE ) != 0 )
        std::wcout << sax::bg::wred << L"out-of-memory" << nl;

    if ( crypto_pwhash_scryptsalsa208sha256_str_verify ( hashed_password, PASSWORD, strlen ( PASSWORD ) ) != 0 )
        std::wcout << sax::fg::wred << L"password incorrect" << nl;
    else
        std::wcout << sax::fg::wgreen << L"password accepted" << nl;

    leveldb::DB * db;
    leveldb::Options options;
    options.create_if_missing = true;
    leveldb::Status status    = leveldb::DB::Open ( options, "y:/metube/leveldb", &db );

    ytid ytid;

    if ( status.ok ( ) ) {
        std::string value;
        leveldb::Status s = db->Get ( leveldb::ReadOptions ( ), "ytid", &value );
        if ( not s.ok ( ) ) {
            std::wcout << sax::fg::wgreen << L"ytid created" << nl;
            status = db->Put ( leveldb::WriteOptions ( ), "ytid", leveldb::Slice ( ytid.data ( ), ytid.size ( ) ) );
        }
        else {
            std::wcout << sax::fg::wred << L"ytid not created" << sax::fg::green << " data: ";
            std::ostringstream data;
            data << std::string{ "data data data data data data data data data data data data data data data data data data data "
                                 "data data data data data" };
            data.flush ( );
            std::cout << sax::fg::bright_blue << data.str ( ) << nl;
        }
    }

    delete db;

    std::wcout << sax::fg::wyellow;

    return EXIT_SUCCESS;
}
