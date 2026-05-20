'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const links = [
  { href: '/',             label: 'Dashboard'    },
  { href: '/simulation',   label: 'Simulação'    },
  { href: '/verification', label: 'Verificação'  },
  { href: '/metodologia',  label: 'Metodologia'  },
];

export default function Nav() {
  const pathname = usePathname();

  return (
    <nav className="bg-gray-800 border-b border-gray-700 sticky top-0 z-50">
      <div className="container mx-auto px-4 max-w-6xl flex items-center gap-1 h-12">
        <span className="text-blue-400 font-bold mr-4 text-sm tracking-wide">
          Cart-Pole DQN
        </span>
        {links.map(({ href, label }) => {
          const active = pathname === href || (href !== '/' && pathname.startsWith(href));
          return (
            <Link
              key={href}
              href={href}
              className={`px-3 py-1.5 rounded text-sm transition-colors ${
                active
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              {label}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
